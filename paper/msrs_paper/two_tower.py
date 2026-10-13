"""ID 없는 Two-Tower.

상품 타워: 상품의 임베딩 입력(텍스트 / 이미지 / 둘을 이어 붙임) → MLP
고객 타워: 고객이 산 상품들의 '상품 타워 입력'을 평균 → MLP
섞기 대응표는 item_features() 한 곳에서만 적용하므로 두 타워에 똑같이 들어간다.

학습 표본: 주 단위. 각 라벨 주의 구매를 정답으로, 그 주 이전 구매를 고객 이력으로 쓴다 (미래 정보 누수 없음).
- tune 모드: 라벨 주가 검증 주 직전에서 끝나고, 검증 주로 채점한다.
- final 모드: 라벨 주에 검증 주까지 포함하고, 테스트 주 후보를 만든다.

인기도 입력 (조건 spec의 pop: true, 추가 분석): 상품 타워 입력에 [직전 1주, 직전 4주 판매량의 log1p를 z-점수화]
2개를 붙인다. 라벨 주마다 그 주 시작 이전 판매량으로 따로 계산하므로 정답 주의 판매가 섞이지 않는다.
고객 타워 입력은 그대로(콘텐츠 임베딩 평균)다.

상품 ID (조건 spec의 id: true, 2차 사전 등록): 상품 타워 입력 = 콘텐츠 ⊕ 학습되는 상품 ID 임베딩.
학습 데이터(해당 모드의 history)에 한 번도 없는 상품의 ID는 예측 때 항상 0이다.
ID 드롭아웃 p: 학습 중 표본마다 확률 p로 ID 부분 전체를 0으로 바꾼다 (배율 보정 없음, 신규 상품 상태를 흉내).
고객 타워는 그대로. ID를 쓰지 않는 조건은 이 기능이 추가되기 전과 계산 경로가 같다.
"""
import time

import numpy as np
import pandas as pd

from . import metrics
from .config import load_yaml, train_params
from .data_prep import load_split
from .utils import environment_info, local_tmp_dir, now, publish_dir, read_json, set_seed, write_json


def item_features(paths, spec, seed):
    parts = []
    if spec["text"]:
        parts.append(np.load(paths.embeddings / "text_minilm.npy"))
    if spec["image"]:
        img = np.load(paths.embeddings / f"img_{spec['image']}.npy")
        if spec["shuffle"]:
            perm = np.load(paths.embeddings / f"shuffle_{spec['shuffle']}_seed{seed}.npy")
            img = img[perm]
        parts.append(img)
    return np.concatenate(parts, axis=1).astype(np.float32)


def user_history(rows, users, max_hist):
    """users 순서대로, 최근 구매 순 중복 없는 상품 max_hist개 (flat, offsets)."""
    r = rows[rows["user"].isin(users)].sort_values(["user", "t_dat"], ascending=[True, False])
    r = r.drop_duplicates(["user", "item_idx"])
    r = r[r.groupby("user").cumcount() < max_hist]
    pos = pd.Series(np.arange(len(users)), index=users)
    key = pos.loc[r["user"].values].values
    order = np.argsort(key, kind="stable")
    flat = r["item_idx"].values[order].astype(np.int64)
    counts = np.bincount(key, minlength=len(users))
    offsets = np.concatenate([[0], np.cumsum(counts)[:-1]]).astype(np.int64)
    return flat, offsets, counts


def pop_features(history, starts, n_items, windows=(1, 4)):
    """(len(starts), n_items, len(windows)). 각 시작 시점 이전 판매량의 log1p를 z-점수화."""
    out = np.zeros((len(starts), n_items, len(windows)), dtype=np.float32)
    t = history["t_dat"]
    for i, s in enumerate(starts):
        for j, weeks in enumerate(windows):
            m = (t >= s - pd.Timedelta(weeks=weeks)) & (t < s)
            v = np.log1p(np.bincount(history.loc[m, "item_idx"].values, minlength=n_items)).astype(np.float32)
            out[i, :, j] = (v - v.mean()) / (v.std() + 1e-6)
    return out


def build_samples(history, period_end, label_weeks, max_hist):
    flats, offs, pair_key, pair_item, pair_week, week_starts = [], [], [], [], [], []
    n_keys, n_flat = 0, 0
    for k in range(label_weeks):
        wk_end = period_end - pd.Timedelta(weeks=k)
        wk_start = wk_end - pd.Timedelta(weeks=1)
        lab = history[(history["t_dat"] >= wk_start) & (history["t_dat"] < wk_end)][["user", "item_idx"]]
        lab = lab.drop_duplicates()
        prior = history[history["t_dat"] < wk_start]
        users = np.intersect1d(lab["user"].unique(), prior["user"].unique())
        flat, off, _ = user_history(prior, users, max_hist)
        lab = lab[lab["user"].isin(users)]
        pos = pd.Series(np.arange(len(users)), index=users)
        pair_key.append(pos.loc[lab["user"].values].values + n_keys)
        pair_item.append(lab["item_idx"].values)
        pair_week.append(np.full(len(lab), k))
        week_starts.append(wk_start)
        flats.append(flat)
        offs.append(off + n_flat)
        n_keys += len(users)
        n_flat += len(flat)
    return {"flat": np.concatenate(flats), "offsets": np.concatenate(offs),
            "pair_key": np.concatenate(pair_key).astype(np.int64),
            "pair_item": np.concatenate(pair_item).astype(np.int64),
            "pair_week": np.concatenate(pair_week).astype(np.int64), "week_starts": week_starts}


def mean_features(Xt, flat, offsets, device, chunk=200_000):
    import torch
    import torch.nn.functional as F
    flat_t = torch.from_numpy(flat).to(device)
    out = []
    for s in range(0, len(offsets), chunk):
        e = min(len(offsets), s + chunk)
        lo, hi = offsets[s], (offsets[e] if e < len(offsets) else len(flat))
        off = torch.from_numpy(offsets[s:e] - lo).to(device)
        out.append(F.embedding_bag(flat_t[lo:hi], Xt, off, mode="mean"))
    return torch.cat(out)


def build_model(d_user, d_item, p):
    import torch.nn as nn

    def mlp(d_in):
        return nn.Sequential(nn.Linear(d_in, p["hidden"]), nn.ReLU(), nn.Dropout(p["dropout"]),
                             nn.Linear(p["hidden"], p["emb_dim"]))

    model = nn.Module()
    model.user, model.item = mlp(d_user), mlp(d_item)
    return model


def id_settings(cfg, spec, mode, hp_name):
    """ID 조건의 {id_dim, id_dropout}와 설정 이름. ID를 쓰지 않으면 (None, None)."""
    if not spec.get("id"):
        return None, None
    candidates = load_yaml("prereg_v2")["id_model"]["tuning"]["candidates"]
    if mode == "tune":
        name = hp_name
    elif spec.get("id_setting"):
        name = spec["id_setting"]
    else:
        assert cfg["id_model"]["tuned"] and cfg["id_model"]["selected"], \
            "ID 튜닝 결과를 paper.yaml id_model.selected에 적고 tuned: true로 커밋한 뒤 실행하세요"
        name = cfg["id_model"]["selected"]
    return dict(candidates[name]), name


def train(X, samples, p, seed, device, log=print, P=None, id_cfg=None):
    """P: 라벨 주별 인기도 입력 (label_weeks, n_items, k) 또는 None. id_cfg: {id_dim, id_dropout} 또는 None."""
    import torch
    import torch.nn.functional as F
    set_seed(seed)
    Xt = torch.from_numpy(X).to(device)
    U = mean_features(Xt, samples["flat"], samples["offsets"], device)
    keys = torch.from_numpy(samples["pair_key"]).to(device)
    items = torch.from_numpy(samples["pair_item"]).to(device)
    weeks = torch.from_numpy(samples["pair_week"]).to(device)
    Pt = None if P is None else torch.from_numpy(P).to(device)
    d_item = X.shape[1] + (0 if P is None else P.shape[2]) + (0 if id_cfg is None else id_cfg["id_dim"])
    model = build_model(X.shape[1], d_item, p)
    if id_cfg is not None:
        model.id_emb = torch.nn.Embedding(X.shape[0], id_cfg["id_dim"])
        torch.nn.init.normal_(model.id_emb.weight, std=0.01)
    model = model.to(device)
    opt = torch.optim.Adam(model.parameters(), lr=p["lr"], weight_decay=p["weight_decay"])
    gen = torch.Generator(device="cpu").manual_seed(seed)
    n, bs = len(keys), p["batch_size"]
    for epoch in range(p["epochs"]):
        model.train()
        order = torch.randperm(n, generator=gen).to(device)
        total = 0.0
        for s in range(0, n, bs):
            b = order[s:s + bs]
            kb, ib = keys[b], items[b]
            item_in = Xt[ib] if Pt is None else torch.cat([Xt[ib], Pt[weeks[b], ib]], dim=1)
            if id_cfg is not None:
                e = model.id_emb(ib)
                if id_cfg["id_dropout"] > 0:
                    e = e * (torch.rand(len(b), 1, device=device) >= id_cfg["id_dropout"]).float()
                item_in = torch.cat([item_in, e], dim=1)
            u = F.normalize(model.user(U[kb]), dim=1)
            v = F.normalize(model.item(item_in), dim=1)
            logits = u @ v.T / p["temperature"]
            same = ib[None, :] == ib[:, None]          # 같은 상품이 배치에 또 있으면 부정 샘플에서 제외
            same.fill_diagonal_(False)
            logits = logits.masked_fill(same, -1e4)
            loss = F.cross_entropy(logits, torch.arange(len(b), device=device))
            opt.zero_grad()
            loss.backward()
            opt.step()
            total += loss.item() * len(b)
        log(f"epoch {epoch + 1}/{p['epochs']} loss {total / n:.4f}")
    return model


def predict(model, X, flat, offsets, k, device, chunk=4096, P_pred=None, seen=None):
    """seen: ID 모델에서 학습 데이터에 등장한 상품 표시 (bool, n_items). 없는 상품의 ID는 0으로 둔다."""
    import torch
    import torch.nn.functional as F
    model.eval()
    with torch.no_grad():
        Xt = torch.from_numpy(X).to(device)
        item_in = Xt if P_pred is None else torch.cat([Xt, torch.from_numpy(P_pred).to(device)], dim=1)
        if hasattr(model, "id_emb"):
            mask = torch.from_numpy(seen.astype(np.float32)).to(device)[:, None]
            item_in = torch.cat([item_in, model.id_emb.weight * mask], dim=1)
        V = F.normalize(model.item(item_in), dim=1)
        Uin = mean_features(Xt, flat, offsets, device)
        out = []
        for s in range(0, len(Uin), chunk):
            u = F.normalize(model.user(Uin[s:s + chunk]), dim=1)
            out.append((u @ V.T).topk(k, dim=1).indices.int().cpu().numpy())
    return np.concatenate(out)


def run(cfg, exps, paths, cond, seed, split, group, mode="final", hp_name=None, device=None, log=print,
        train_window_weeks=None, out_subdir=None):
    """실험 1회. 드라이브에 run_info.json이 이미 있으면 건너뛴다 (중단 후 이어 돌리기 가능).

    train_window_weeks: None이면 전체 학습 기간 사용(기존 동작). 정수면 테스트/검증 시작 직전 그 주 수만큼만 history로 사용.
    out_subdir: None이면 기존 경로. 지정하면 outputs/{group}/{split}/{out_subdir}/{cond}_seed{seed}에 저장.
    """
    import torch
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    if mode == "final":
        name = f"{cond}_seed{seed}"
        if out_subdir:
            out_dir = paths.outputs / group / split / out_subdir / name
        else:
            out_dir = paths.run_dir(group, split, name)
        overrides = None
    else:
        name = f"{cond}_seed{seed}_{hp_name}"
        out_dir = paths.outputs / "tuning" / split / name
        # 학습 설정 후보(h1~h3)면 학습 설정을, ID 후보(i1~i4)면 ID 설정만 바꾼다
        overrides = cfg["train"]["candidates"].get(hp_name)
    if (out_dir / "run_info.json").exists():
        log(f"[skip] {out_dir} 이미 완료")
        return read_json(out_dir / "run_info.json")

    p = train_params(cfg, overrides)
    spec = exps["conditions"][cond]
    id_cfg, id_name = id_settings(cfg, spec, mode, hp_name)
    stats = read_json(paths.split_dir(split) / "stats.json")
    history, valid_targets, test_targets = load_split(paths, split)
    start = pd.Timestamp(stats["valid_start"] if mode == "tune" else stats["test_start"])
    targets = valid_targets if mode == "tune" else test_targets
    history = history[history["t_dat"] < start]
    if train_window_weeks is not None:
        window_start = start - pd.Timedelta(weeks=int(train_window_weeks))
        history = history[history["t_dat"] >= window_start]
        log(f"[{name}] train_window_weeks={train_window_weeks}: history {len(history):,}건 (기간 {window_start.date()} ~ {start.date()})")

    t0 = time.time()
    X = item_features(paths, spec, seed)
    samples = build_samples(history, start, p["label_weeks"], p["max_hist"])
    log(f"[{name}] {split}/{mode} 입력 {X.shape}, 학습 쌍 {len(samples['pair_key']):,}, device {device}")
    P = P_pred = None
    if spec.get("pop"):
        P = pop_features(history, samples["week_starts"], X.shape[0])
        P_pred = pop_features(history, [start], X.shape[0])[0]
    model = train(X, samples, p, seed, device, log, P=P, id_cfg=id_cfg)
    users = np.sort(targets["user"].unique())
    flat, offsets, counts = user_history(history, users, p["max_hist"])
    seen = np.bincount(history["item_idx"].values, minlength=X.shape[0]) > 0
    cands = predict(model, X, flat, offsets, cfg["eval"]["k"], device, P_pred=P_pred, seen=seen)

    info = {"name": name, "cond": cond, "spec": spec, "seed": seed, "split": split, "group": group, "mode": mode,
            "hp_name": hp_name, "params": p, "n_items": int(X.shape[0]),
            "input_dim": int(X.shape[1]) + (0 if P is None else P.shape[2]) + (0 if id_cfg is None else id_cfg["id_dim"]),
            "id_setting": id_name, "id_cfg": id_cfg, "n_seen_items": int(seen.sum()),
            "n_train_pairs": int(len(samples["pair_key"])), "n_users": int(len(users)),
            "users_without_history": int((counts == 0).sum()), "train_seconds": round(time.time() - t0, 1),
            "train_window_weeks": train_window_weeks, "out_subdir": out_subdir,
            "created_at": now(), **environment_info()}
    if mode == "tune":
        info["valid_recall@300"] = metrics.recall_user_mean(cands, users, targets, cfg["eval"]["k"])
        log(f"[{name}] valid Recall@300 = {info['valid_recall@300']:.4f}")

    tmp = local_tmp_dir()
    np.save(tmp / "candidates.npy", cands.astype(np.int32))
    np.save(tmp / "users.npy", users.astype(np.int64))
    write_json(tmp / "run_info.json", info)
    publish_dir(tmp, out_dir)
    log(f"[{name}] 저장 → {out_dir}")
    return info


def run_group(cfg, exps, paths, group, log=print):
    for block in exps["groups"][group]:
        for seed in block["seeds"]:
            for cond in block["conds"]:
                run(cfg, exps, paths, cond, seed, block["split"], group, log=log)
