"""가설 B 분석: 상품 수명 주기 단계별 상대 이미지 기여도.

사전등록(prereg_B_lifecycle.yaml):
- 라벨: processed/{split}/lifecycle_labels.parquet (scripts/label_lifecycle.py 산출물)
- 학습 산출물: outputs/W/main/w26w/{C2,C3}_seed{42,43,44} (가설 A 재사용)
- 주 세그먼트: non_repurchase 안에서 lifecycle 단계로 서브셋 분할
- 주 검정: 상대 기여도(growth) − 상대 기여도(maturity), 짝지은 95% CI 하한 > 0 이면 지지
- 보조: (1) growth − 0.5(maturity + decline), (2) min(intro,growth,decline) − maturity

처리 일관성(가설 A와 동일 원칙):
- 모든 조건·seed·단계가 같은 users 목록·같은 부트스트랩 추출을 공유
- 세그먼트·단계 필터는 집계 전에 적용
- seed별 상대 기여도 평균

사용 (cwd=paper/):
  python -m scripts.analysis_lifecycle \\
    --drive-root /content/drive/MyDrive/msrs_paper \\
    --prereg paper/configs/prereg_B_lifecycle.yaml \\
    --out /content/drive/MyDrive/msrs_paper/results/B_lifecycle_results.json
"""
import argparse
import json
import sys
from itertools import combinations
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
    """고객별 Recall@K 분자(hits)와 분모(n). analysis_window._r_vector_per_user와 동일."""
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
    """가중된 상대 기여도: [R_r − R_c] / R_c (고객별 평균 Recall)."""
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


def _abs_contribution(h_r, n_r, h_c, n_c, w):
    def um(h, n, w):
        has = (n > 0).astype(np.float64)
        r = np.divide(h, n, out=np.zeros_like(h, dtype=np.float64), where=n > 0)
        denom = (w * has).sum()
        return float((w * r * has).sum() / denom) if denom > 0 else np.nan
    return um(h_r, n_r, w) - um(h_c, n_c, w)


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
    p.add_argument("--labels", default=None, help="기본값: processed/{split}/lifecycle_labels.parquet")
    p.add_argument("--out", required=True)
    args = p.parse_args()

    metrics, Paths = _load()
    paths = Paths(Path(args.drive_root))
    prereg = yaml.safe_load(Path(args.prereg).read_text(encoding="utf-8"))

    seeds = prereg["measure"]["seeds"]
    n_boot = prereg["measure"]["bootstrap"]["n"]
    boot_seed = prereg["measure"]["bootstrap"]["seed"]
    real_cond = prereg["runs"]["conditions"]["real"]
    ctrl_cond = prereg["runs"]["conditions"]["shuffle_global"]
    group = prereg["runs"]["group"]
    split = prereg["runs"]["split"]
    subdir = prereg["runs"]["out_subdir"]
    stages = prereg["segments"]["stages"]
    all_stages = list(stages) + ["unassigned"]
    primary_segment = "non_repurchase"
    k = 300

    # 1) 테스트 정답
    test = pd.read_parquet(paths.split_dir(split) / "test_targets.parquet")
    print(f"테스트 구매: {len(test):,}건")

    # 2) 라벨
    labels_path = Path(args.labels) if args.labels else (paths.split_dir(split) / "lifecycle_labels.parquet")
    labels = pd.read_parquet(labels_path)
    print(f"라벨 상품: {len(labels):,} (분포: {labels['stage'].value_counts().to_dict()})")

    stage_of_item = pd.Series(labels["stage"].values, index=labels["item_idx"].values)
    test_stage = test["item_idx"].map(stage_of_item).fillna("unassigned")
    # history에 없는 상품(테스트 신규 상품)은 is_new_item 기반으로 introduction으로 보정
    new_item_mask = test["is_new_item"].values & (test_stage == "unassigned").values
    test = test.copy()
    test["stage"] = test_stage.values
    test.loc[new_item_mask, "stage"] = "introduction"

    # 3) 세그먼트·단계 교차 마스크
    non_repurchase = ~test["is_repurchase"].values
    masks = {}
    for stg in all_stages:
        m = non_repurchase & (test["stage"].values == stg)
        masks[("non_repurchase", stg)] = m
        print(f"  non_repurchase × {stg:>12}: {int(m.sum()):>6,}건 구매")
    # 보조: all × stage
    for stg in all_stages:
        m = test["stage"].values == stg
        masks[("all", stg)] = m

    # 4) 모든 실행의 후보 로드 (가설 A의 w26w 재사용)
    print(f"\n실행 로드: {subdir} × 2 conds × {len(seeds)} seeds")
    runs = {}
    users_ref = None
    for cond in [real_cond, ctrl_cond]:
        for s in seeds:
            r = _load_run(paths.outputs, group, split, subdir, cond, s)
            if r is None:
                raise FileNotFoundError(f"{group}/{split}/{subdir}/{cond}_seed{s}: 가설 A의 산출물이 없다. 가설 A를 먼저 실행하거나 train_window로 window=26만 추가 실행하라.")
            cands, users, d = r
            if users_ref is None:
                users_ref = users
            elif not np.array_equal(users, users_ref):
                raise ValueError(f"users 불일치: {d}")
            runs[(cond, s)] = (cands, users, d)
    print(f"로드 완료: {len(runs)} 실행")

    # 5) (segment, stage)별 rows·items 사전 계산
    seg_data = {}
    for key, mask in masks.items():
        t_seg = test[mask].reset_index(drop=True)
        if len(t_seg) == 0:
            print(f"[빈 서브셋] {key} — 건너뜀")
            continue
        rows = metrics.target_rows(users_ref, t_seg)
        items_idx = t_seg["item_idx"].values
        seg_data[key] = (rows, items_idx, int(len(t_seg)))

    # 6) 각 (cond, seed, segment, stage)별 고객별 hits·n
    print("\n채점 중...")
    tallies = {}
    for (cond, s), (cands, _, _) in runs.items():
        for key, (rows, items_idx, _) in seg_data.items():
            h_u, n_u = _r_vector_per_user(cands, users_ref, rows, items_idx, k)
            tallies[(cond, s, key)] = (h_u, n_u)

    # 7) 점추정: (segment, stage)별 seed 평균 상대 기여도
    U = len(users_ref)
    w_ones = np.ones(U, dtype=np.float64)
    point_rel = {}
    point_abs = {}
    for key in seg_data:
        rel_vals, abs_vals = [], []
        for s in seeds:
            tr = tallies.get((real_cond, s, key))
            tc = tallies.get((ctrl_cond, s, key))
            if tr is None or tc is None:
                continue
            rel_vals.append(_rel_contribution(tr[0], tr[1], tc[0], tc[1], w_ones))
            abs_vals.append(_abs_contribution(tr[0], tr[1], tc[0], tc[1], w_ones))
        point_rel[key] = float(np.nanmean(rel_vals)) if rel_vals else np.nan
        point_abs[key] = float(np.nanmean(abs_vals)) if abs_vals else np.nan

    # 8) 부트스트랩 (짝지은: 모든 조건·단계가 같은 추출 공유)
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

    # 9) 결과 집계
    def pack(key):
        pt = point_rel.get(key, np.nan)
        boots = boot_rel.get(key, [])
        lo, hi = _q(boots, 0.95)
        return {
            "estimate_rel": pt,
            "ci_low": lo, "ci_high": hi,
            "estimate_abs": point_abs.get(key, np.nan),
            "n_purchases": seg_data.get(key, (None, None, 0))[2],
        }

    by_segment = {}
    for seg in ("non_repurchase", "all"):
        by_segment[seg] = {stg: pack((seg, stg)) for stg in all_stages if (seg, stg) in seg_data}

    # 10) 확증 검정
    def diff_ci(key_a, key_b):
        a = np.array(boot_rel.get(key_a, []))
        b = np.array(boot_rel.get(key_b, []))
        if len(a) == 0 or len(b) == 0:
            return {"estimate": np.nan, "ci_low": np.nan, "ci_high": np.nan, "supported": False}
        d = a - b
        pt = point_rel[key_a] - point_rel[key_b]
        lo, hi = _q(d, 0.95)
        return {"estimate": float(pt), "ci_low": lo, "ci_high": hi, "supported": bool(lo > 0)}

    primary = diff_ci(("non_repurchase", "growth"), ("non_repurchase", "maturity"))
    primary["comparison"] = "상대 기여도(growth) − 상대 기여도(maturity) in non_repurchase"

    # 보조 1: growth − 0.5(maturity + decline)
    a = np.array(boot_rel.get(("non_repurchase", "growth"), []))
    bm = np.array(boot_rel.get(("non_repurchase", "maturity"), []))
    bd = np.array(boot_rel.get(("non_repurchase", "decline"), []))
    if len(a) and len(bm) and len(bd):
        d1 = a - 0.5 * (bm + bd)
        pt1 = point_rel[("non_repurchase", "growth")] - 0.5 * (point_rel[("non_repurchase", "maturity")] + point_rel[("non_repurchase", "decline")])
        lo1, hi1 = _q(d1, 0.95)
        sec1 = {
            "comparison": "상대 기여도(growth) − 0.5 × [상대 기여도(maturity) + 상대 기여도(decline)]",
            "estimate": float(pt1), "ci_low": lo1, "ci_high": hi1, "supported": bool(lo1 > 0),
        }
    else:
        sec1 = {"comparison": "...", "estimate": np.nan, "ci_low": np.nan, "ci_high": np.nan, "supported": False}

    # 보조 2: min(intro, growth, decline) − maturity (짝지은)
    bi = np.array(boot_rel.get(("non_repurchase", "introduction"), []))
    if len(bi) and len(a) and len(bd) and len(bm):
        min_other = np.minimum.reduce([bi, a, bd])
        d2 = min_other - bm
        min_other_pt = min(point_rel[("non_repurchase", "introduction")],
                           point_rel[("non_repurchase", "growth")],
                           point_rel[("non_repurchase", "decline")])
        pt2 = min_other_pt - point_rel[("non_repurchase", "maturity")]
        lo2, hi2 = _q(d2, 0.95)
        sec2 = {
            "comparison": "min(상대 기여도(introduction, growth, decline)) − 상대 기여도(maturity)",
            "estimate": float(pt2), "ci_low": lo2, "ci_high": hi2, "supported": bool(lo2 > 0),
        }
    else:
        sec2 = {"comparison": "...", "estimate": np.nan, "ci_low": np.nan, "ci_high": np.nan, "supported": False}

    # 11) 탐색적: 모든 쌍대 비교
    pairwise = {}
    for a_name, b_name in combinations(stages, 2):
        pair_key = f"{a_name}_vs_{b_name}"
        pairwise[pair_key] = diff_ci(("non_repurchase", a_name), ("non_repurchase", b_name))
        pairwise[pair_key]["comparison"] = f"상대 기여도({a_name}) − 상대 기여도({b_name}) in non_repurchase"

    # 12) 라벨링 메타 (상품 단계 분포 등)
    labels_meta = {
        "n_items_labeled": int(len(labels)),
        "stage_counts": {k: int(v) for k, v in labels["stage"].value_counts().to_dict().items()},
        "stage_mean_total_sales": labels.groupby("stage")["total_sales"].mean().to_dict(),
        "stage_mean_age_weeks": labels.groupby("stage")["age_weeks"].mean().to_dict(),
    }

    results = {
        "kind": "B_lifecycle_analysis",
        "created_by": "paper/scripts/analysis_lifecycle.py",
        "prereg": args.prereg,
        "labels_file": str(labels_path),
        "seeds": seeds,
        "n_boot": n_boot,
        "boot_seed": boot_seed,
        "n_users": int(U),
        "by_segment": by_segment,
        "confirmatory_primary": primary,
        "confirmatory_secondary": [sec1, sec2],
        "exploratory_pairwise_non_repurchase": pairwise,
        "labels_meta": labels_meta,
    }

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(results, indent=2, ensure_ascii=False, default=float), encoding="utf-8")

    print("\n=== 요약 ===")
    print(f"\n[단계별 상대 이미지 기여도] (non_repurchase, seed 3개 평균, 95% CI)")
    for stg in all_stages:
        d = by_segment.get("non_repurchase", {}).get(stg)
        if d is None:
            continue
        print(f"  {stg:>12}: {d['estimate_rel']:+.4f} [{d['ci_low']:+.4f}, {d['ci_high']:+.4f}]  n={d['n_purchases']:,}")
    c = primary
    print(f"\n[주 검정] growth − maturity = {c['estimate']:+.4f} [{c['ci_low']:+.4f}, {c['ci_high']:+.4f}]  지지={c['supported']}")
    print(f"[보조 1] {sec1['comparison']}")
    print(f"         = {sec1['estimate']:+.4f} [{sec1['ci_low']:+.4f}, {sec1['ci_high']:+.4f}]  지지={sec1['supported']}")
    print(f"[보조 2] {sec2['comparison']}")
    print(f"         = {sec2['estimate']:+.4f} [{sec2['ci_low']:+.4f}, {sec2['ci_high']:+.4f}]  지지={sec2['supported']}")
    print(f"\n저장: {args.out}")


if __name__ == "__main__":
    main()
