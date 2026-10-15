"""가설 C1 분석: 학습 window × 상품 수명 주기 단계 교호작용.

사전등록: paper/configs/prereg_C1_window_x_lifecycle.yaml

입력:
  - outputs/W/main/w{window}w/{C2,C3}_seed{seed}/candidates.npy (가설 A)
  - processed/main/lifecycle_labels.parquet (가설 B)

출력: results/C1_window_x_lifecycle_results.json

사용 (cwd=paper/):
  python -m scripts.analysis_window_x_lifecycle \\
    --drive-root /content/drive/MyDrive/msrs_paper \\
    --prereg paper/configs/prereg_C1_window_x_lifecycle.yaml \\
    --out /content/drive/MyDrive/msrs_paper/results/C1_window_x_lifecycle_results.json
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
    p.add_argument("--labels", default=None, help="기본값: processed/{split}/lifecycle_labels.parquet")
    p.add_argument("--out", required=True)
    args = p.parse_args()

    metrics, Paths = _load()
    paths = Paths(Path(args.drive_root))
    prereg = yaml.safe_load(Path(args.prereg).read_text(encoding="utf-8"))

    windows = prereg["inputs"]["windows"]
    seeds = prereg["inputs"]["seeds"]
    real_cond = prereg["inputs"]["conditions"]["real"]
    ctrl_cond = prereg["inputs"]["conditions"]["shuffle_global"]
    stages = prereg["inputs"]["stages"]
    all_stages = list(stages) + ["unassigned"]
    group, split = "W", "main"
    k = 300
    n_boot = prereg["measure"]["bootstrap"]["n"]
    boot_seed = prereg["measure"]["bootstrap"]["seed"]
    primary_segment = "non_repurchase"

    # 1) 테스트 정답
    test = pd.read_parquet(paths.split_dir(split) / "test_targets.parquet")
    print(f"테스트 구매: {len(test):,}건")

    # 2) 라벨 (가설 B 산출물)
    labels_path = Path(args.labels) if args.labels else (paths.split_dir(split) / "lifecycle_labels.parquet")
    if not labels_path.exists():
        raise FileNotFoundError(
            f"{labels_path} 없음. 가설 B 노트북의 라벨링 셀을 먼저 돌려야 함 "
            f"(python -m scripts.label_lifecycle ...)."
        )
    labels = pd.read_parquet(labels_path)
    print(f"라벨 상품: {len(labels):,} (분포: {labels['stage'].value_counts().to_dict()})")

    stage_of_item = pd.Series(labels["stage"].values, index=labels["item_idx"].values)
    test_stage = test["item_idx"].map(stage_of_item).fillna("unassigned")
    new_item_mask = test["is_new_item"].values & (test_stage == "unassigned").values
    test = test.copy()
    test["stage"] = test_stage.values
    test.loc[new_item_mask, "stage"] = "introduction"

    # 3) 세그먼트 × 단계 마스크
    non_repurchase = ~test["is_repurchase"].values
    masks = {}
    for stg in all_stages:
        masks[("non_repurchase", stg)] = non_repurchase & (test["stage"].values == stg)
        masks[("all", stg)] = test["stage"].values == stg

    # 4) 모든 (window, cond, seed) 산출물 로드
    print(f"\n실행 로드: {len(windows)} windows × 2 conds × {len(seeds)} seeds = {len(windows)*2*len(seeds)}회")
    runs = {}
    users_ref = None
    for w in windows:
        subdir = f"w{w}w"
        for cond in [real_cond, ctrl_cond]:
            for s in seeds:
                r = _load_run(paths.outputs, group, split, subdir, cond, s)
                if r is None:
                    raise FileNotFoundError(f"{group}/{split}/{subdir}/{cond}_seed{s}: 가설 A 산출물 없음.")
                cands, users, d = r
                if users_ref is None:
                    users_ref = users
                elif not np.array_equal(users, users_ref):
                    raise ValueError(f"users 불일치: {d}")
                runs[(w, cond, s)] = (cands, users, d)
    print(f"로드 완료: {len(runs)} 실행")

    # 5) (segment, stage)별 rows·items 사전 계산
    seg_data = {}
    for key, mask in masks.items():
        t_seg = test[mask].reset_index(drop=True)
        if len(t_seg) == 0:
            continue
        rows = metrics.target_rows(users_ref, t_seg)
        items_idx = t_seg["item_idx"].values
        seg_data[key] = (rows, items_idx, int(len(t_seg)))

    # 6) 모든 (window, cond, seed, segment, stage) 조합 채점
    print("\n채점 중...")
    tallies = {}
    for (w, cond, s), (cands, _, _) in runs.items():
        for seg_key, (rows, items_idx, _) in seg_data.items():
            h_u, n_u = _r_vector_per_user(cands, users_ref, rows, items_idx, k)
            tallies[(w, cond, s, seg_key)] = (h_u, n_u)

    # 7) 점추정: (window, segment, stage)별 seed 평균 상대 기여도
    U = len(users_ref)
    w_ones = np.ones(U, dtype=np.float64)
    point_rel = {}   # (window, segment, stage) → rel
    for w in windows:
        for seg_key in seg_data:
            vals = []
            for s in seeds:
                tr = tallies.get((w, real_cond, s, seg_key))
                tc = tallies.get((w, ctrl_cond, s, seg_key))
                if tr is None or tc is None:
                    continue
                vals.append(_rel_contribution(tr[0], tr[1], tc[0], tc[1], w_ones))
            point_rel[(w, seg_key)] = float(np.nanmean(vals)) if vals else np.nan

    # 8) 짝지은 부트스트랩 (모든 window × cond × stage × seed가 같은 추출 공유)
    print(f"\n부트스트랩 {n_boot}회...")
    boot_rel = {key: [] for key in point_rel}
    for it, w_boot in enumerate(metrics.bootstrap_weights(U, n_boot, boot_seed)):
        w_boot = w_boot.astype(np.float64)
        for w in windows:
            for seg_key in seg_data:
                vals = []
                for s in seeds:
                    tr = tallies.get((w, real_cond, s, seg_key))
                    tc = tallies.get((w, ctrl_cond, s, seg_key))
                    if tr is None or tc is None:
                        continue
                    vals.append(_rel_contribution(tr[0], tr[1], tc[0], tc[1], w_boot))
                boot_rel[(w, seg_key)].append(float(np.nanmean(vals)) if vals else np.nan)
        if (it + 1) % 200 == 0:
            print(f"  {it + 1}/{n_boot}")

    # 9) by_window × by_stage 집계
    def pack(w, seg_key):
        pt = point_rel.get((w, seg_key), np.nan)
        boots = boot_rel.get((w, seg_key), [])
        lo, hi = _q(boots, 0.95)
        return {"estimate_rel": pt, "ci_low": lo, "ci_high": hi,
                "n_purchases": seg_data.get(seg_key, (None, None, 0))[2]}

    by_window_stage = {}
    for w in windows:
        by_window_stage[str(w)] = {}
        for seg_name in ("non_repurchase", "all"):
            by_window_stage[str(w)][seg_name] = {
                stg: pack(w, (seg_name, stg))
                for stg in all_stages if (seg_name, stg) in seg_data
            }

    # 10) 확증 검정
    w_min, w_max = min(windows), max(windows)   # 보통 2 vs 26
    # Δ_s = R_rel(w=min, s) − R_rel(w=max, s), stage s ∈ stages
    delta_point = {}
    delta_boot = {}
    for stg in stages:
        key_lo = (w_min, ("non_repurchase", stg))
        key_hi = (w_max, ("non_repurchase", stg))
        delta_point[stg] = point_rel[key_lo] - point_rel[key_hi]
        a = np.array(boot_rel[key_lo])
        b = np.array(boot_rel[key_hi])
        delta_boot[stg] = a - b

    # 주 검정: Δ_maturity − Δ_introduction
    diff_mat_intro = delta_point["maturity"] - delta_point["introduction"]
    diff_mat_intro_boot = delta_boot["maturity"] - delta_boot["introduction"]
    lo, hi = _q(diff_mat_intro_boot, 0.95)
    primary = {
        "comparison": "[Δ_maturity − Δ_introduction]  where Δ_s = R_rel(w={}, s) − R_rel(w={}, s)".format(w_min, w_max),
        "estimate": float(diff_mat_intro),
        "ci_low": lo, "ci_high": hi,
        "supported": bool(lo > 0),
    }

    # 보조 1: 각 단계 Δ_s CI
    stage_replication = {}
    for stg in stages:
        lo_s, hi_s = _q(delta_boot[stg], 0.95)
        stage_replication[stg] = {
            "delta_estimate": float(delta_point[stg]),
            "ci_low": lo_s, "ci_high": hi_s,
            "supported": bool(lo_s > 0),
        }

    # 보조 2: Δ_maturity − max(Δ_intro, Δ_growth, Δ_decline)
    others = ["introduction", "growth", "decline"]
    others_point = max(delta_point[s] for s in others)
    # 부트스트랩에서 "max 다른 세 단계"는 iteration별 max
    others_boot = np.max(np.vstack([delta_boot[s] for s in others]), axis=0)
    diff_mat_max_other = delta_point["maturity"] - others_point
    diff_mat_max_other_boot = delta_boot["maturity"] - others_boot
    lo2, hi2 = _q(diff_mat_max_other_boot, 0.95)
    secondary_maturity_max = {
        "comparison": "Δ_maturity − max(Δ_introduction, Δ_growth, Δ_decline)",
        "estimate": float(diff_mat_max_other),
        "ci_low": lo2, "ci_high": hi2,
        "supported": bool(lo2 > 0),
    }

    # 11) 탐색적: 선형 추세 β_s (각 단계)
    xs = np.log2(np.array(windows, dtype=np.float64))
    xm = xs - xs.mean()
    denom = float((xm * xm).sum())
    beta_by_stage = {}
    for stg in stages:
        y_pt = np.array([point_rel[(w, ("non_repurchase", stg))] for w in windows])
        ym = y_pt - y_pt.mean()
        beta_point = float((xm * ym).sum() / denom)
        betas = []
        for it in range(n_boot):
            y_boot = np.array([boot_rel[(w, ("non_repurchase", stg))][it] for w in windows])
            if np.all(np.isfinite(y_boot)):
                ym_b = y_boot - y_boot.mean()
                betas.append((xm * ym_b).sum() / denom)
        lo_b, hi_b = _q(betas, 0.95)
        beta_by_stage[stg] = {
            "beta_estimate": beta_point,
            "ci_low": lo_b, "ci_high": hi_b,
            "supported_neg": bool(hi_b < 0),
        }

    # 12) 탐색적: 6쌍의 Δ_s − Δ_s' (교호작용 지도)
    pairwise_delta = {}
    for a, b in combinations(stages, 2):
        pair = f"{a}_vs_{b}"
        diff = delta_point[a] - delta_point[b]
        diff_b = delta_boot[a] - delta_boot[b]
        lo_p, hi_p = _q(diff_b, 0.95)
        pairwise_delta[pair] = {
            "comparison": f"Δ_{a} − Δ_{b}",
            "estimate": float(diff),
            "ci_low": lo_p, "ci_high": hi_p,
        }

    # 13) 결과 저장
    results = {
        "kind": "C1_window_x_lifecycle_analysis",
        "created_by": "paper/scripts/analysis_window_x_lifecycle.py",
        "prereg": args.prereg,
        "labels_file": str(labels_path),
        "windows": windows,
        "seeds": seeds,
        "n_boot": n_boot,
        "boot_seed": boot_seed,
        "n_users": int(U),
        "by_window_stage": by_window_stage,
        "confirmatory_primary": primary,
        "secondary_stage_replication": stage_replication,
        "secondary_maturity_max": secondary_maturity_max,
        "exploratory_linear_trend_per_stage": beta_by_stage,
        "exploratory_pairwise_delta": pairwise_delta,
    }
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(results, indent=2, ensure_ascii=False, default=float), encoding="utf-8")

    # 14) 요약 출력
    print("\n=== 요약 ===")
    print(f"\n[window × stage 상대 기여도] (non_repurchase, seed 평균, 95% CI)")
    header = "window  " + "  ".join(f"{s:>12}" for s in stages)
    print(header)
    for w in windows:
        row = f"{w:>6}  "
        for stg in stages:
            d = by_window_stage[str(w)]["non_repurchase"].get(stg)
            if d is None:
                row += f"{'—':>12}  "
            else:
                row += f"{d['estimate_rel']:+.3f} ({d['ci_low']:+.3f},{d['ci_high']:+.3f}) "
        print(row)

    print(f"\n[단계별 window 효과 Δ_s = R_rel(w={w_min}) − R_rel(w={w_max})]")
    for stg in stages:
        d = stage_replication[stg]
        print(f"  {stg:>12}: {d['delta_estimate']:+.4f} [{d['ci_low']:+.4f}, {d['ci_high']:+.4f}]  지지={d['supported']}")

    c = primary
    print(f"\n[주 검정] Δ_maturity − Δ_introduction")
    print(f"  = {c['estimate']:+.4f} [{c['ci_low']:+.4f}, {c['ci_high']:+.4f}]  지지={c['supported']}")
    sm = secondary_maturity_max
    print(f"\n[보조] {sm['comparison']}")
    print(f"  = {sm['estimate']:+.4f} [{sm['ci_low']:+.4f}, {sm['ci_high']:+.4f}]  지지={sm['supported']}")

    print(f"\n저장: {args.out}")


if __name__ == "__main__":
    main()
