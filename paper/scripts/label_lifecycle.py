"""가설 B: 상품 수명 주기 단계 라벨링.

사전등록: paper/configs/prereg_B_lifecycle.yaml

입력: processed/{split}/history.parquet (train+valid 거래)
출력: processed/{split}/lifecycle_labels.parquet
  컬럼: item_idx, stage, first_week, age_weeks, total_sales, recent_sales,
        recent_slope, peak_week_offset, meets_floor

사용 (cwd=paper/):
  python -m scripts.label_lifecycle \\
    --drive-root /content/drive/MyDrive/msrs_paper \\
    --prereg paper/configs/prereg_B_lifecycle.yaml \\
    --split main
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
    from msrs_paper import data_prep
    from msrs_paper.config import Paths
    return data_prep, Paths


def _weekly_sales_matrix(history, split_dates):
    """상품 × 주 행렬. 열은 테스트 주 시작(test_start)으로부터의 역산 주차.

    week_idx = (t_dat − train_start) // 7 (정수). 0 = 학습 1주차.
    반환: items (sorted item_idx list), M (I × W) int32, train_start, n_weeks
    """
    train_start = pd.Timestamp(split_dates["train_start"])
    test_start = pd.Timestamp(split_dates["test_start"])
    valid_start = pd.Timestamp(split_dates["valid_start"])

    h = history[history["t_dat"] < test_start].copy()
    week_idx = ((h["t_dat"] - train_start).dt.days // 7).astype(np.int32)
    h["week_idx"] = week_idx

    n_weeks = int(((test_start - train_start).days + 6) // 7)
    items = np.sort(h["item_idx"].unique()).astype(np.int32)
    item_pos = pd.Series(np.arange(len(items), dtype=np.int32), index=items)

    rows = item_pos.loc[h["item_idx"].values].values
    cols = np.clip(h["week_idx"].values, 0, n_weeks - 1)
    counts = np.bincount(rows * n_weeks + cols, minlength=len(items) * n_weeks).reshape(len(items), n_weeks)
    return items, counts.astype(np.int32), train_start, n_weeks, valid_start, test_start


def _recent_slope(mat, recent_weeks):
    """각 행에 대해 log1p(weekly_sales) ~ week_offset 선형 회귀 β. 최근 recent_weeks 주만 사용.

    week_offset = 0 (가장 오래된 참조 주) → recent_weeks-1 (테스트 직전 주).
    분모는 Var(x) × recent_weeks, 모든 상품 동일.
    """
    recent = mat[:, -recent_weeks:].astype(np.float64)
    y = np.log1p(recent)
    x = np.arange(recent_weeks, dtype=np.float64)
    xm = x - x.mean()
    denom = float((xm * xm).sum())
    y_mean = y.mean(axis=1, keepdims=True)
    ym = y - y_mean
    beta = (ym * xm).sum(axis=1) / denom
    return beta


def _peak_week_offset(mat):
    """peak가 테스트 직전으로부터 얼마나 '옛날'인지(주 단위). 0 = 테스트 직전 주.

    peak가 복수면 가장 최근(가장 작은 offset)을 쓴다.
    """
    rev = mat[:, ::-1]   # 오른쪽부터 (가장 최근부터) 보도록 뒤집음
    peak_rev = rev.argmax(axis=1)   # 가장 최근 peak의 역순 index
    return peak_rev.astype(np.int32)


def _label(features, rule):
    """우선순위 priority 순서대로 룰을 적용해 라벨을 매긴다."""
    n = len(features)
    stage = np.full(n, "unassigned", dtype=object)
    assigned = np.zeros(n, dtype=bool)

    conds = {
        "introduction": features["age_weeks"] <= rule["intro_weeks"],
        "decline": (
            (features["recent_slope"] <= rule["slope_down"])
            & (features["peak_week_offset"] >= rule["recent_weeks"])
            & (features["recent_sales"] >= rule["recent_sales_floor"])
        ),
        "growth": (
            (features["recent_slope"] >= rule["slope_up"])
            & (features["recent_sales"] >= rule["recent_sales_floor"])
            & (features["age_weeks"] > rule["intro_weeks"])
        ),
        "maturity": (
            (features["recent_slope"].abs() <= rule["flat_threshold"])
            & (features["recent_sales"] >= rule["recent_sales_floor"])
            & (features["total_sales"] >= rule["maturity_cutoff"])
        ),
    }
    for name in rule["priority"]:
        if name == "unassigned":
            continue
        mask = conds[name].values & ~assigned
        stage[mask] = name
        assigned |= mask
    return stage


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--drive-root", required=True)
    p.add_argument("--prereg", required=True)
    p.add_argument("--split", default="main")
    p.add_argument("--out", default=None, help="기본값: processed/{split}/lifecycle_labels.parquet")
    args = p.parse_args()

    data_prep, Paths = _load()
    paths = Paths(Path(args.drive_root))
    prereg = yaml.safe_load(Path(args.prereg).read_text(encoding="utf-8"))
    rule = prereg["labeling"]

    split_dir = paths.split_dir(args.split)
    history = pd.read_parquet(split_dir / "history.parquet")
    stats = json.loads((split_dir / "stats.json").read_text(encoding="utf-8"))
    split_dates = {k: stats[k] for k in ("train_start", "valid_start", "test_start", "end")}

    items, mat, train_start, n_weeks, valid_start, test_start = _weekly_sales_matrix(history, split_dates)
    print(f"history 상품 수: {len(items):,}, 참조 주 수: {n_weeks}")
    print(f"학습 시작: {train_start.date()}  → 테스트 시작: {test_start.date()}")

    # 상품별 특징
    first_week = (mat > 0).argmax(axis=1)       # 처음 등장한 주 인덱스 (0부터)
    sold_ever = (mat.sum(axis=1) > 0)
    first_week = np.where(sold_ever, first_week, n_weeks)   # 안 팔린 상품은 가짜 큰 값
    age_weeks = n_weeks - first_week             # 등장 이후 지난 주 수
    total_sales = mat.sum(axis=1)
    recent_sales = mat[:, -rule["recent_weeks"]:].sum(axis=1)
    recent_slope = _recent_slope(mat, rule["recent_weeks"])
    peak_week_offset = _peak_week_offset(mat)

    # 성숙 분위수 (학습 기간 누적 판매). sold_ever 상품만 고려해 분위 계산
    sold_total = total_sales[sold_ever]
    if len(sold_total) == 0:
        maturity_cutoff = np.inf
    else:
        maturity_cutoff = float(np.quantile(sold_total, rule["maturity_cumulative_quantile"]))
    print(f"성숙 커트라인 Q({rule['maturity_cumulative_quantile']}): total_sales ≥ {maturity_cutoff:.1f}")

    df = pd.DataFrame({
        "item_idx": items,
        "first_week": first_week.astype(np.int32),
        "age_weeks": age_weeks.astype(np.int32),
        "total_sales": total_sales.astype(np.int32),
        "recent_sales": recent_sales.astype(np.int32),
        "recent_slope": recent_slope.astype(np.float32),
        "peak_week_offset": peak_week_offset.astype(np.int32),
    })
    rule_eff = dict(rule)
    rule_eff["maturity_cutoff"] = maturity_cutoff
    df["stage"] = _label(df, rule_eff)

    # 저장
    out = Path(args.out) if args.out else (split_dir / "lifecycle_labels.parquet")
    df.to_parquet(out, index=False)

    # 메타 (재현·감사용)
    meta = {
        "kind": "lifecycle_labels",
        "prereg": args.prereg,
        "split": args.split,
        "train_start": str(train_start.date()),
        "test_start": str(test_start.date()),
        "n_weeks": int(n_weeks),
        "n_items_in_history": int(len(items)),
        "maturity_cutoff_total_sales": maturity_cutoff,
        "stage_counts": {k: int(v) for k, v in df["stage"].value_counts().to_dict().items()},
        "rule": {k: (v if not isinstance(v, (np.floating,)) else float(v)) for k, v in rule.items()},
    }
    meta_path = out.with_suffix(".meta.json")
    meta_path.write_text(json.dumps(meta, indent=2, ensure_ascii=False, default=float), encoding="utf-8")

    # 요약 출력
    print("\n=== 라벨 분포 (history에 등장한 상품만) ===")
    for name, cnt in df["stage"].value_counts().items():
        print(f"  {name:>12}: {cnt:>7,}  ({cnt/len(df):.1%})")
    print(f"\n저장: {out}")
    print(f"메타: {meta_path}")


if __name__ == "__main__":
    main()
