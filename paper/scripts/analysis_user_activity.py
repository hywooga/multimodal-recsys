"""가설 C2 분석: 사용자 활동량 분위수별 상대 이미지 기여도.

사전등록: paper/configs/prereg_C2_user_activity.yaml

입력:
  - processed/{split}/user_activity_labels.parquet (scripts/label_user_activity.py 산출물)
  - outputs/W/main/w26w/{C2,C3}_seed{42,43,44} (가설 A 재사용)

사용 (cwd=paper/):
  python -m scripts.analysis_user_activity \\
    --drive-root /content/drive/MyDrive/msrs_paper \\
    --prereg paper/configs/prereg_C2_user_activity.yaml \\
    --out /content/drive/MyDrive/msrs_paper/results/C2_user_activity_results.json
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
    from msrs_paper.config import Paths
    return metrics, Paths


def _r_vector_per_user(cands, users_ref, rows, items_idx, k):
    from msrs_paper.metrics import hits_at_k
    hit = hits_at_k(cands, rows, items_idx, k).astype(np.float32)
    U = len(users_ref)
    n_u = np.bincount(rows, minlength=U).astype(np.float32)
    h_u = np.bincount(rows, weights=hit, minlength=U).astype(np.float32)
    return h_u, n_u


def _rel_contribution(h_r, n_r, h_c, n_c, w):
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
    p.add_argument("--labels", default=None, help="기본값: processed/{split}/user_activity_labels.parquet")
    p.add_argument("--out", required=True)
    args = p.parse_args()

    metrics, Paths = _load()
    paths = Paths(Path(args.drive_root))
    prereg = yaml.safe_load(Path(args.prereg).read_text(encoding="utf-8"))

    seeds = prereg["measure"]["seeds"]
    n_boot = prereg["measure"]["bootstrap"]["n"]
    boot_seed = prereg["measure"]["bootstrap"]["seed"]
    real_cond = "C2"
    ctrl_cond = "C3"
    group = prereg["runs"]["group"]
    split = prereg["runs"]["split"]
    subdir = prereg["runs"]["out_subdir"]
    k = 300
    primary_segment = "non_repurchase"

    # 1) 테스트 정답
    test = pd.read_parquet(paths.split_dir(split) / "test_targets.parquet")
    print(f"테스트 구매: {len(test):,}건")

    # 2) 라벨 (가설 C2 label_user_activity 산출물)
    labels_path = Path(args.labels) if args.labels else (paths.split_dir(split) / "user_activity_labels.parquet")
    if not labels_path.exists():
        raise FileNotFoundError(
            f"{labels_path} 없음. 먼저 python -m scripts.label_user_activity 실행."
        )
    labels = pd.read_parquet(labels_path)
    print(f"라벨 사용자: {len(labels):,}, 분위수 분포:\n{labels['activity_quintile'].value_counts().sort_index()}")

    # 3) 테스트 구매에 분위수 join
    quint_of_user = pd.Series(labels["activity_quintile"].values, index=labels["user"].values)
    test = test.copy()
    test["activity_quintile"] = test["user"].map(quint_of_user)
    missing = test["activity_quintile"].isna().sum()
    if missing:
        print(f"[주의] 라벨 없는 테스트 구매 {missing:,}건 — 분석에서 제외")
    test = test[~test["activity_quintile"].isna()].copy()
    test["activity_quintile"] = test["activity_quintile"].astype(int)

    quintiles = sorted(test["activity_quintile"].unique().tolist())
    print(f"분위수: {quintiles}")

    # 4) 세그먼트 × 분위수 마스크
    non_repurchase = ~test["is_repurchase"].values
    masks = {}
    for q in quintiles:
        masks[("non_repurchase", q)] = non_repurchase & (test["activity_quintile"].values == q)
        masks[("all", q)] = test["activity_quintile"].values == q
        print(f"  non_repurchase × Q{q}: {int(masks[('non_repurchase', q)].sum()):>6,}건 구매")

    # 5) 가설 A의 w26 산출물 로드
    runs = {}
    users_ref = None
    for cond in [real_cond, ctrl_cond]:
        for s in seeds:
            d = paths.outputs / group / split / subdir / f"{cond}_seed{s}"
            if not (d / "candidates.npy").exists():
                raise FileNotFoundError(f"{d}: 가설 A의 w26w 산출물 없음.")
            cands = np.load(d / "candidates.npy")
            users = np.load(d / "users.npy")
            if users_ref is None:
                users_ref = users
            elif not np.array_equal(users, users_ref):
                raise ValueError(f"users 불일치: {d}")
            runs[(cond, s)] = (cands, users, d)
    print(f"로드 완료: {len(runs)} 실행")

    # 6) 세그먼트별 rows·items
    seg_data = {}
    for key, mask in masks.items():
        t_seg = test[mask].reset_index(drop=True)
        if len(t_seg) == 0:
            continue
        rows = metrics.target_rows(users_ref, t_seg)
        items_idx = t_seg["item_idx"].values
        seg_data[key] = (rows, items_idx, int(len(t_seg)))

    # 7) 채점
    print("\n채점 중...")
    tallies = {}
    for (cond, s), (cands, _, _) in runs.items():
        for key, (rows, items_idx, _) in seg_data.items():
            h_u, n_u = _r_vector_per_user(cands, users_ref, rows, items_idx, k)
            tallies[(cond, s, key)] = (h_u, n_u)

    # 8) 점추정
    U = len(users_ref)
    w_ones = np.ones(U, dtype=np.float64)
    point_rel = {}
    for key in seg_data:
        vals = []
        for s in seeds:
            tr = tallies.get((real_cond, s, key))
            tc = tallies.get((ctrl_cond, s, key))
            if tr is None or tc is None:
                continue
            vals.append(_rel_contribution(tr[0], tr[1], tc[0], tc[1], w_ones))
        point_rel[key] = float(np.nanmean(vals)) if vals else np.nan

    # 9) 짝지은 부트스트랩
    print(f"\n부트스트랩 {n_boot}회...")
    boot_rel = {key: [] for key in seg_data}
    for it, w_boot in enumerate(metrics.bootstrap_weights(U, n_boot, boot_seed)):
        w_boot = w_boot.astype(np.float64)
        for key in seg_data:
            vals = []
            for s in seeds:
                tr = tallies.get((real_cond, s, key))
                tc = tallies.get((ctrl_cond, s, key))
                if tr is None or tc is None:
                    continue
                vals.append(_rel_contribution(tr[0], tr[1], tc[0], tc[1], w_boot))
            boot_rel[key].append(float(np.nanmean(vals)) if vals else np.nan)
        if (it + 1) % 200 == 0:
            print(f"  {it + 1}/{n_boot}")

    # 10) 집계
    def pack(key):
        pt = point_rel.get(key, np.nan)
        boots = boot_rel.get(key, [])
        lo, hi = _q(boots, 0.95)
        return {"estimate_rel": pt, "ci_low": lo, "ci_high": hi,
                "n_purchases": seg_data.get(key, (None, None, 0))[2]}

    by_segment = {}
    for seg in ("non_repurchase", "all"):
        by_segment[seg] = {f"Q{q}": pack((seg, q)) for q in quintiles if (seg, q) in seg_data}

    # 11) 주 검정: Q1 − Q5 (CI 하한 > 0 이면 지지)
    q_min, q_max = min(quintiles), max(quintiles)
    diff_point = point_rel[("non_repurchase", q_min)] - point_rel[("non_repurchase", q_max)]
    diff_boot = np.array(boot_rel[("non_repurchase", q_min)]) - np.array(boot_rel[("non_repurchase", q_max)])
    lo, hi = _q(diff_boot, 0.95)
    primary = {
        "comparison": f"상대 기여도(Q{q_min}, 저활동) − 상대 기여도(Q{q_max}, 고활동) in non_repurchase",
        "estimate": float(diff_point),
        "ci_low": lo, "ci_high": hi,
        "supported": bool(lo > 0),
    }

    # 12) 보조: 선형 추세 β (분위수 index → 상대 기여도)
    xs = np.array(quintiles, dtype=np.float64)
    xm = xs - xs.mean()
    denom = float((xm * xm).sum())
    y_pt = np.array([point_rel[("non_repurchase", q)] for q in quintiles])
    ym = y_pt - y_pt.mean()
    beta_point = float((xm * ym).sum() / denom)
    betas = []
    for it in range(n_boot):
        y_boot = np.array([boot_rel[("non_repurchase", q)][it] for q in quintiles])
        if np.all(np.isfinite(y_boot)):
            ym_b = y_boot - y_boot.mean()
            betas.append((xm * ym_b).sum() / denom)
    lo_b, hi_b = _q(betas, 0.95)
    secondary_trend = {
        "regressor": "activity_quintile (1~5)",
        "beta_estimate": beta_point,
        "ci_low": lo_b, "ci_high": hi_b,
        "supported_neg": bool(hi_b < 0),   # 음의 추세: 활동량 ↑ → 기여도 ↓
    }

    # 13) 라벨 메타 (분위수별 활동량 요약)
    label_meta_path = labels_path.with_suffix(".meta.json")
    label_meta = json.loads(label_meta_path.read_text(encoding="utf-8")) if label_meta_path.exists() else {}

    results = {
        "kind": "C2_user_activity_analysis",
        "created_by": "paper/scripts/analysis_user_activity.py",
        "prereg": args.prereg,
        "labels_file": str(labels_path),
        "seeds": seeds,
        "n_boot": n_boot,
        "boot_seed": boot_seed,
        "n_users": int(U),
        "n_users_labeled": int(len(labels)),
        "quintiles": quintiles,
        "by_segment": by_segment,
        "confirmatory_primary": primary,
        "secondary_linear_trend": secondary_trend,
        "label_meta": label_meta,
    }
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(results, indent=2, ensure_ascii=False, default=float), encoding="utf-8")

    print("\n=== 요약 ===")
    print(f"\n[분위수별 상대 이미지 기여도] (non_repurchase, seed 평균, 95% CI)")
    for q in quintiles:
        d = by_segment["non_repurchase"].get(f"Q{q}")
        print(f"  Q{q}: {d['estimate_rel']:+.4f} [{d['ci_low']:+.4f}, {d['ci_high']:+.4f}]  n={d['n_purchases']:,}")
    c = primary
    print(f"\n[주 검정] {c['comparison']}")
    print(f"  = {c['estimate']:+.4f} [{c['ci_low']:+.4f}, {c['ci_high']:+.4f}]  지지={c['supported']}")
    t = secondary_trend
    print(f"\n[보조] 선형 추세 β({t['regressor']}) = {t['beta_estimate']:+.4f} [{t['ci_low']:+.4f}, {t['ci_high']:+.4f}]  음의 추세 지지={t['supported_neg']}")
    print(f"\n저장: {args.out}")


if __name__ == "__main__":
    main()
