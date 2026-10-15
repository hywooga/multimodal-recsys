"""가설 C2: 사용자 활동량(train 기간 구매 수) 분위수 라벨링.

사전등록: paper/configs/prereg_C2_user_activity.yaml

입력: processed/{split}/history.parquet, stats.json
출력: processed/{split}/user_activity_labels.parquet
  컬럼: user, train_purchases, activity_quintile (1~5, 1=최저)

사용 (cwd=paper/):
  python -m scripts.label_user_activity \\
    --drive-root /content/drive/MyDrive/msrs_paper \\
    --prereg paper/configs/prereg_C2_user_activity.yaml \\
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
    from msrs_paper.config import Paths
    return Paths


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--drive-root", required=True)
    p.add_argument("--prereg", required=True)
    p.add_argument("--split", default="main")
    p.add_argument("--out", default=None)
    args = p.parse_args()

    Paths = _load()
    paths = Paths(Path(args.drive_root))
    prereg = yaml.safe_load(Path(args.prereg).read_text(encoding="utf-8"))
    quantiles = prereg["labeling"]["quantiles"]

    split_dir = paths.split_dir(args.split)
    history = pd.read_parquet(split_dir / "history.parquet")
    stats = json.loads((split_dir / "stats.json").read_text(encoding="utf-8"))
    valid_start = pd.Timestamp(stats["valid_start"])

    # train 구간만 집계 (사전등록에 명시)
    train = history[history["t_dat"] < valid_start]
    print(f"train 거래: {len(train):,}건 (valid_start={valid_start.date()} 이전)")

    counts = train.groupby("user").size().rename("train_purchases").reset_index()
    print(f"사용자 수: {len(counts):,}")

    # 분위수 라벨 (1=최저, Q 수=최상위). duplicates='drop'으로 동점 처리
    try:
        counts["activity_quintile"] = pd.qcut(
            counts["train_purchases"], q=quantiles, labels=False, duplicates="drop"
        ) + 1
    except ValueError as e:
        raise RuntimeError(f"qcut 실패 (분위수 축소 필요?): {e}")

    actual_q = int(counts["activity_quintile"].max())
    if actual_q < quantiles:
        print(f"[주의] 동점으로 실제 분위수 수가 {quantiles} → {actual_q} 로 축소됨 (사전등록 명시).")

    out = Path(args.out) if args.out else (split_dir / "user_activity_labels.parquet")
    counts.to_parquet(out, index=False)

    # 요약
    summary = counts.groupby("activity_quintile")["train_purchases"].agg(
        n_users="count", mean="mean", median="median", min="min", max="max"
    ).round(1)
    print("\n=== 분위수별 활동량 요약 ===")
    print(summary)

    meta = {
        "kind": "user_activity_labels",
        "prereg": args.prereg,
        "split": args.split,
        "reference_range_end": str(valid_start.date()),
        "n_users": int(len(counts)),
        "n_train_purchases_total": int(len(train)),
        "requested_quantiles": int(quantiles),
        "actual_quantiles": int(actual_q),
        "summary_per_quintile": summary.reset_index().to_dict(orient="records"),
    }
    meta_path = out.with_suffix(".meta.json")
    meta_path.write_text(json.dumps(meta, indent=2, ensure_ascii=False, default=float), encoding="utf-8")

    print(f"\n저장: {out}")
    print(f"메타: {meta_path}")


if __name__ == "__main__":
    main()
