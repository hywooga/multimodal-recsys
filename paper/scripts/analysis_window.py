"""가설 A 분석: window별 상대 이미지 기여도.

사전등록(prereg_A_window.yaml):
- 주 검정: non_repurchase 구간에서 상대 기여도(window=2) − 상대 기여도(window=26). 짝지은 부트스트랩 95% CI 하한 > 0 이면 지지
- 보조: 선형 추세 β < 0 (log2(window)에 대한 회귀)
- 보고 구간: non_repurchase(주), all·new_item(탐색적)

처리 일관성 (sensitivity_labels.py와 동일 원칙):
- 모든 window·cond·seed가 **같은 users 목록 · 같은 test_targets · 같은 부트스트랩 추출**을 공유
- 셀별 집계는 segment 필터링 후 수행
- seed별 상대 기여도 계산 후 평균

사용 (cwd=paper/):
  python -m scripts.analysis_window \\
    --drive-root /content/drive/MyDrive/msrs_paper \\
    --prereg /content/repo/paper/configs/prereg_A_window.yaml \\
    --out /content/drive/MyDrive/msrs_paper/results/A_window_results.json
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import yaml


def _load():
    here = Path(__file__).resolve()
    sys.path.insert(0, str(here.parent.parent))
    from msrs_paper import metrics
    from msrs_paper.config import Paths, load_yaml
    return metrics, Paths, load_yaml


def _r_vector_per_user(cands, users_ref, rows, items_idx, k):
    """고객별 Recall@K 분자(hits)와 분모(n). 집계 후 bootstrap에 쓸 (hits, n)."""
    from msrs_paper.metrics import hits_at_k
    hit = hits_at_k(cands, rows, items_idx, k).astype(np.float32)
    U = len(users_ref)
    n_u = np.bincount(rows, minlength=U).astype(np.float32)
    h_u = np.bincount(rows, weights=hit, minlength=U).astype(np.float32)
    return h_u, n_u


def _load_run(outputs, group, split, subdir, cond, seed):
    d = outputs / group / split / subdir / f"{cond}_seed{seed}"
    if not (d / "candidates.npy").exists():
        return None
    return np.load(d / "candidates.npy"), np.load(d / "users.npy"), d


def _rel_contribution(h_r, n_r, h_c, n_c, w):
    """가중된 상대 기여도: [R_r - R_c] / R_c, 고객별 평균 Recall."""
    def um(h, n, w):
        has = (n > 0).astype(np.float64)
        r = np.divide(h, n, out=np.zeros_like(h, dtype=np.float64), where=n > 0)
        denom = (w * has).sum()
        return float((w * r * has).sum() / denom) if denom > 0 else np.nan
    R_r = um(h_r, n_r, w)
    R_c = um(h_c, n_c, w)
    if R_c <= 0 or not np.isfinite(R_c):
        return np.nan
    return (R_r - R_c) / R_c


def _q(x, lvl):
    x = np.asarray(x, dtype=np.float64)
    x = x[np.isfinite(x)]
    if len(x) == 0:
        return float("nan"), float("nan")
    a = (1 - lvl) / 2
    return float(np.quantile(x, a)), float(np.quantile(x, 1 - a))


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--drive-root", required=True)
    p.add_argument("--prereg", required=True)
    p.add_argument("--out", required=True)
    args = p.parse_args()

    metrics, Paths, load_yaml = _load()
    paths = Paths(Path(args.drive_root))
    prereg = yaml.safe_load(Path(args.prereg).read_text(encoding="utf-8"))

    windows = prereg["windows"]
    seeds = prereg["measure"]["seeds"]
    real_cond = prereg["conditions"]["real"]
    ctrl_cond = prereg["conditions"]["shuffle_global"]
    group = prereg["runs"]["group"]
    split = prereg["runs"]["split"]
    k = 300
    n_boot = prereg["measure"]["bootstrap"]["n"]
    boot_seed = prereg["measure"]["bootstrap"]["seed"]
    primary_segment = "non_repurchase"
    all_segments = [primary_segment, "all", "new_item"]

    # 1) 테스트 정답 로드
    test = pd.read_parquet(paths.split_dir(split) / "test_targets.parquet")
    print(f"테스트 구매: {len(test):,}건")

    # 2) 세그먼트 마스크
    seg_masks = {
        "all": np.ones(len(test), dtype=bool),
        "non_repurchase": ~test["is_repurchase"].values,
        "new_item": test["is_new_item"].values,
    }

    # 3) 모든 실행의 후보 로드
    print(f"\n실행 로드: {len(windows)} windows × 2 conds × {len(seeds)} seeds")
    runs = {}   # (window, cond, seed) → (cands, users, dir)
    users_ref = None
    for w in windows:
        subdir = f"w{w}w"
        for cond in [real_cond, ctrl_cond]:
            for s in seeds:
                r = _load_run(paths.outputs, group, split, subdir, cond, s)
                if r is None:
                    print(f"[missing] window={w} cond={cond} seed={s}")
                    continue
                cands, users, d = r
                if users_ref is None:
                    users_ref = users
                elif not np.array_equal(users, users_ref):
                    raise ValueError(f"users 불일치: {d}")
                runs[(w, cond, s)] = (cands, users, d)
    print(f"로드 완료: {len(runs)} 실행")

    # 4) 세그먼트별 rows·items 사전 계산
    seg_data = {}
    for seg_name, mask in seg_masks.items():
        t_seg = test[mask].reset_index(drop=True)
        rows = metrics.target_rows(users_ref, t_seg)
        items_idx = t_seg["item_idx"].values
        seg_data[seg_name] = (rows, items_idx, t_seg)
        print(f"segment={seg_name}: {len(t_seg):,}건 구매")

    # 5) 모든 (window, cond, seed, segment)에 대해 고객별 hits·n 계산
    print("\n채점 중...")
    tallies = {}   # (window, cond, seed, segment) → (h_u, n_u) 둘 다 (U,)
    for key, (cands, users, _) in runs.items():
        w, cond, s = key
        for seg_name, (rows, items_idx, _) in seg_data.items():
            h_u, n_u = _r_vector_per_user(cands, users_ref, rows, items_idx, k)
            tallies[(w, cond, s, seg_name)] = (h_u, n_u)

    # 6) 점추정: seed별 상대 기여도 → 평균
    U = len(users_ref)
    w_ones = np.ones(U, dtype=np.float64)
    point_rel = {}   # (window, segment) → seed 평균 상대 기여도
    for w in windows:
        for seg_name in all_segments:
            seed_vals = []
            for s in seeds:
                tr = tallies.get((w, real_cond, s, seg_name))
                tc = tallies.get((w, ctrl_cond, s, seg_name))
                if tr is None or tc is None:
                    continue
                seed_vals.append(_rel_contribution(tr[0], tr[1], tc[0], tc[1], w_ones))
            point_rel[(w, seg_name)] = float(np.nanmean(seed_vals)) if seed_vals else np.nan

    # 7) 부트스트랩 (짝지은: 모든 조건이 같은 w 공유)
    print(f"\n부트스트랩 {n_boot}회...")
    boot_rel = {k2: [] for k2 in point_rel}
    for it, w_boot in enumerate(metrics.bootstrap_weights(U, n_boot, boot_seed)):
        w_boot = w_boot.astype(np.float64)
        for w in windows:
            for seg_name in all_segments:
                seed_vals = []
                for s in seeds:
                    tr = tallies.get((w, real_cond, s, seg_name))
                    tc = tallies.get((w, ctrl_cond, s, seg_name))
                    if tr is None or tc is None:
                        continue
                    seed_vals.append(_rel_contribution(tr[0], tr[1], tc[0], tc[1], w_boot))
                boot_rel[(w, seg_name)].append(float(np.nanmean(seed_vals)) if seed_vals else np.nan)
        if (it + 1) % 200 == 0:
            print(f"  {it + 1}/{n_boot}")

    # 8) 결과 집계
    def pack(w, seg):
        pt = point_rel.get((w, seg), np.nan)
        boots = boot_rel.get((w, seg), [])
        lo, hi = _q(boots, 0.95)
        return {"estimate": pt, "ci_low": lo, "ci_high": hi}

    results = {
        "kind": "A_window_analysis",
        "created_by": "paper/scripts/analysis_window.py",
        "prereg": "paper/configs/prereg_A_window.yaml",
        "windows": windows,
        "seeds": seeds,
        "n_boot": n_boot,
        "boot_seed": boot_seed,
        "n_users": U,
        "n_test_purchases": {s: int(seg_masks[s].sum()) for s in all_segments},
        "by_window": {
            str(w): {seg: pack(w, seg) for seg in all_segments}
            for w in windows
        },
    }

    # 9) 확증 검정: 극단 비교 (window=2 vs window=26, non_repurchase)
    w_min, w_max = min(windows), max(windows)
    diff_point = point_rel[(w_min, primary_segment)] - point_rel[(w_max, primary_segment)]
    diff_boot = np.array(boot_rel[(w_min, primary_segment)]) - np.array(boot_rel[(w_max, primary_segment)])
    lo, hi = _q(diff_boot, 0.95)
    results["confirmatory_primary"] = {
        "comparison": f"상대 기여도(window={w_min}) − 상대 기여도(window={w_max}) in {primary_segment}",
        "estimate": float(diff_point),
        "ci_low": lo, "ci_high": hi, "ci_level": 0.95,
        "supported": bool(lo > 0),
    }

    # 10) 보조 검정: 선형 추세 β = Cov(log2(w), y) / Var(log2(w))
    xs = np.log2(np.array(windows, dtype=np.float64))
    y_point = np.array([point_rel[(w, primary_segment)] for w in windows])
    # 점추정 β
    xm = xs - xs.mean()
    ym = y_point - y_point.mean()
    beta_point = float((xm * ym).sum() / (xm * xm).sum())
    # 부트스트랩 β
    betas = []
    for it in range(n_boot):
        y_boot = np.array([boot_rel[(w, primary_segment)][it] for w in windows])
        ym_b = y_boot - y_boot.mean()
        if np.all(np.isfinite(y_boot)):
            betas.append((xm * ym_b).sum() / (xm * xm).sum())
    lo_b, hi_b = _q(betas, 0.95)
    results["confirmatory_secondary_linear_trend"] = {
        "regressor": "log2(window)",
        "beta_estimate": beta_point,
        "beta_ci_low": lo_b, "beta_ci_high": hi_b, "ci_level": 0.95,
        "supported": bool(hi_b < 0),   # 음의 추세 (window 짧을수록 상대 기여도 큼)
    }

    # 11) 보조 정보: 각 run의 n_train_pairs와 train_seconds (수렴 확인)
    train_meta = []
    for key, (_, _, d) in runs.items():
        info = json.loads((d / "run_info.json").read_text(encoding="utf-8"))
        train_meta.append({
            "window": key[0], "cond": key[1], "seed": key[2],
            "n_train_pairs": info.get("n_train_pairs"),
            "n_users": info.get("n_users"),
            "n_seen_items": info.get("n_seen_items"),
            "train_seconds": info.get("train_seconds"),
            "label_weeks": (info.get("params") or {}).get("label_weeks"),
        })
    results["train_meta"] = train_meta

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(results, indent=2, ensure_ascii=False), encoding="utf-8")

    print("\n=== 요약 ===")
    print(f"\n[상대 이미지 기여도] (non_repurchase, seed 3개 평균, 95% CI)")
    for w in windows:
        d = results["by_window"][str(w)][primary_segment]
        print(f"  window={w:>2}주: {d['estimate']:+.4f} [{d['ci_low']:+.4f}, {d['ci_high']:+.4f}]")
    c = results["confirmatory_primary"]
    print(f"\n[주 검정] window={w_min} − window={w_max} = {c['estimate']:+.4f} [{c['ci_low']:+.4f}, {c['ci_high']:+.4f}]  지지={c['supported']}")
    t = results["confirmatory_secondary_linear_trend"]
    print(f"[보조] 선형 추세 β(log2 window) = {t['beta_estimate']:+.4f} [{t['beta_ci_low']:+.4f}, {t['beta_ci_high']:+.4f}]  지지={t['supported']}")
    print(f"\n저장: {args.out}")


if __name__ == "__main__":
    main()
