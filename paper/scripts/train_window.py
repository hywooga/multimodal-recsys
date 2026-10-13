"""가설 A: 학습 window 길이와 이미지 기여도. two_tower.run() 반복 호출.

사전등록: paper/configs/prereg_A_window.yaml
저장 위치: outputs/W/main/w{window}w/{cond}_seed{seed}/

사용 (cwd=paper/):
  python -m scripts.train_window \\
    --drive-root /content/drive/MyDrive/msrs_paper \\
    --windows 2,4,8,16,26 \\
    --conds C0,C2,C3 \\
    --seeds 42,43,44

조정 사항:
- window < label_weeks(6)이면 label_weeks를 min(label_weeks, window)로 자동 조정 (사전등록에 명시)
- 다른 하이퍼파라미터는 paper.yaml train.default(h2) 그대로
"""
import argparse
import sys
from pathlib import Path

import yaml


def _load():
    here = Path(__file__).resolve()
    sys.path.insert(0, str(here.parent.parent))
    from msrs_paper import two_tower
    from msrs_paper.config import Paths, load_yaml
    return two_tower, load_yaml, Paths


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--drive-root", required=True)
    p.add_argument("--windows", default="2,4,8,16,26", help="콤마 구분, 주 단위")
    p.add_argument("--conds", default="C0,C2,C3", help="콤마 구분 조건")
    p.add_argument("--seeds", default="42,43,44", help="콤마 구분 seed")
    p.add_argument("--split", default="main")
    p.add_argument("--group", default="W")
    args = p.parse_args()

    two_tower, load_yaml, Paths = _load()
    paths = Paths(Path(args.drive_root))
    cfg = load_yaml("paper")
    exps = load_yaml("experiments")

    windows = [int(w) for w in args.windows.split(",")]
    conds = [c.strip() for c in args.conds.split(",")]
    seeds = [int(s) for s in args.seeds.split(",")]
    base_label_weeks = cfg["train"]["default"]["label_weeks"]

    print(f"총 실행 수: {len(windows)} × {len(conds)} × {len(seeds)} = {len(windows)*len(conds)*len(seeds)}회")

    for w in windows:
        # window < label_weeks이면 label_weeks 자동 조정 (사전등록 명시)
        eff_label_weeks = min(base_label_weeks, w)
        if eff_label_weeks != base_label_weeks:
            # 임시 cfg 복제 후 수정
            cfg_w = {k: dict(v) if isinstance(v, dict) else v for k, v in cfg.items()}
            cfg_w["train"] = {k: (dict(v) if isinstance(v, dict) else v) for k, v in cfg["train"].items()}
            cfg_w["train"]["default"] = {**cfg["train"]["default"], "label_weeks": eff_label_weeks}
            print(f"[window={w}주] label_weeks={base_label_weeks}→{eff_label_weeks} 자동 조정")
        else:
            cfg_w = cfg

        subdir = f"w{w}w"
        for cond in conds:
            for seed in seeds:
                print(f"\n>>> window={w}w · cond={cond} · seed={seed}")
                try:
                    two_tower.run(cfg_w, exps, paths, cond, seed, args.split, args.group,
                                  mode="final", train_window_weeks=w, out_subdir=subdir)
                except Exception as e:
                    print(f"[실패] window={w} cond={cond} seed={seed}: {e}")

    print("\n완료.")


if __name__ == "__main__":
    main()
