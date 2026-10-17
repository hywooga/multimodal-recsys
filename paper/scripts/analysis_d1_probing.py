"""가설 D1: 상품 임베딩의 속성 선형 분리 가능성 (Representation Probing).

사전등록: paper/configs/prereg_D1_probing.yaml

입력:
  - embeddings/text_minilm.npy, embeddings/img_marqo.npy, embeddings/shuffle_global_seed42.npy
  - processed/items.parquet

출력: results/D1_probing_results.json

사용 (cwd=paper/):
  python -m scripts.analysis_d1_probing \\
    --drive-root /content/drive/MyDrive/msrs_paper \\
    --prereg paper/configs/prereg_D1_probing.yaml \\
    --out /content/drive/MyDrive/msrs_paper/results/D1_probing_results.json
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import yaml
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score, f1_score


def _load():
    here = Path(__file__).resolve()
    sys.path.insert(0, str(here.parent.parent))
    from msrs_paper.config import Paths
    return Paths


def _load_embeddings(paths, prereg):
    """사전등록된 네 임베딩 로드. combined는 concat으로 생성.

    shuffle_global_seed{s}.npy는 2D 임베딩이 아니라 **1D 매핑 인덱스**
    (상품 i 자리에 들어갈 다른 상품의 idx)로 저장되어 있다. 1D일 경우
    먼저 로드된 image 임베딩을 사용해 2D로 재구성한다.
    """
    emb = {}
    for e in prereg["embeddings"]:
        name = e["name"]
        if name == "combined":
            continue   # 뒤에서 생성
        src = paths.embeddings / Path(e["source"]).name
        arr = np.load(src)
        if arr.ndim == 1:
            if "image" not in emb:
                raise RuntimeError(f"{name}이 1D 매핑인데 image 임베딩이 먼저 로드돼 있어야 한다")
            idx = arr.astype(np.int64)
            arr = emb["image"][idx].astype(np.float32)
            print(f"  {name:>16}: {arr.shape} (1D 매핑 → image 임베딩 재구성)")
        else:
            arr = arr.astype(np.float32)
            print(f"  {name:>16}: {arr.shape} from {src.name}")
        emb[name] = arr
    # combined = concat(text, image)
    emb["combined"] = np.concatenate([emb["text"], emb["image"]], axis=1)
    print(f"  {'combined':>16}: {emb['combined'].shape} (concat text + image)")
    return emb


def _prepare_targets(items, prereg):
    """사전등록된 네 target을 라벨 배열로 변환. null 제외, min_class_count 적용."""
    out = {}
    for t in prereg["targets"]:
        col = t["column"]
        y_raw = items[col].values
        mask = pd.notna(y_raw)
        if t.get("exclude_null", False):
            pass   # 이미 mask로 처리
        y_series = pd.Series(y_raw[mask])
        if t["type"] == "binary":
            y = y_series.astype(bool).astype(int).values
            classes = [0, 1]
        else:
            # min_class_count 적용 (소수 클래스 제외)
            min_count = t.get("min_class_count", 1)
            counts = y_series.value_counts()
            keep_classes = counts[counts >= min_count].index.tolist()
            keep_mask = y_series.isin(keep_classes).values
            mask = np.zeros(len(y_raw), dtype=bool)
            mask[np.where(pd.notna(y_raw))[0][keep_mask]] = True
            y_series = y_series[keep_mask]
            # label encoding
            class_to_id = {c: i for i, c in enumerate(sorted(keep_classes))}
            y = np.array([class_to_id[c] for c in y_series.values], dtype=np.int32)
            classes = sorted(keep_classes)
        out[t["name"]] = {
            "mask": mask, "y": y, "classes": classes,
            "n": int(mask.sum()), "n_classes": len(classes),
        }
        print(f"  {t['name']:>24}: n={int(mask.sum()):>6,}, classes={len(classes)}")
    return out


def _probe_one(X, y, cv, clf_params):
    """한 (임베딩, target) 조합에 대해 5-fold CV probing. fold별 예측·정답을 연결하여 반환."""
    y_pred_all = np.zeros_like(y)
    fold_ids = np.zeros_like(y)
    for fold_i, (tr, te) in enumerate(cv.split(X, y)):
        clf = LogisticRegression(**clf_params)
        clf.fit(X[tr], y[tr])
        y_pred_all[te] = clf.predict(X[te])
        fold_ids[te] = fold_i
    acc = float(accuracy_score(y, y_pred_all))
    macro_f1 = float(f1_score(y, y_pred_all, average="macro"))
    per_class_f1 = f1_score(y, y_pred_all, average=None).tolist()
    return {"accuracy": acc, "macro_f1": macro_f1, "per_class_f1": per_class_f1,
            "y": y, "y_pred": y_pred_all, "fold_ids": fold_ids}


def _bootstrap_diff(y, pred_a, pred_b, n_boot, seed):
    """같은 샘플에 대한 두 분류기의 accuracy 차이 (pred_a − pred_b)의 짝지은 부트스트랩."""
    rng = np.random.default_rng(seed)
    n = len(y)
    hits_a = (pred_a == y).astype(np.float64)
    hits_b = (pred_b == y).astype(np.float64)
    diffs = []
    for _ in range(n_boot):
        idx = rng.integers(0, n, n)
        diff = float(hits_a[idx].mean() - hits_b[idx].mean())
        diffs.append(diff)
    pt = float(hits_a.mean() - hits_b.mean())
    lo = float(np.quantile(diffs, 0.025))
    hi = float(np.quantile(diffs, 0.975))
    return {"estimate": pt, "ci_low": lo, "ci_high": hi}


def _bootstrap_acc(y, pred, n_boot, seed):
    """하나의 accuracy에 대한 부트스트랩 CI."""
    rng = np.random.default_rng(seed)
    n = len(y)
    hits = (pred == y).astype(np.float64)
    accs = []
    for _ in range(n_boot):
        idx = rng.integers(0, n, n)
        accs.append(float(hits[idx].mean()))
    pt = float(hits.mean())
    lo = float(np.quantile(accs, 0.025))
    hi = float(np.quantile(accs, 0.975))
    return {"estimate": pt, "ci_low": lo, "ci_high": hi}


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--drive-root", required=True)
    p.add_argument("--prereg", required=True)
    p.add_argument("--out", required=True)
    args = p.parse_args()

    Paths = _load()
    paths = Paths(Path(args.drive_root))
    prereg = yaml.safe_load(Path(args.prereg).read_text(encoding="utf-8"))

    clf_params = prereg["classifier"]["params"]
    cv_params = prereg["cv"]["params"]
    n_boot = prereg["metrics"]["bootstrap_ci"]["n"]
    boot_seed = prereg["metrics"]["bootstrap_ci"]["seed"]

    # 1) items 로드 (상품 수와 임베딩 수가 일치해야 함)
    items = pd.read_parquet(paths.processed / "items.parquet").sort_values("idx").reset_index(drop=True)
    print(f"items.parquet: {len(items):,} 상품")

    # 2) 임베딩 로드
    print("\n임베딩 로드:")
    emb = _load_embeddings(paths, prereg)
    for name, arr in emb.items():
        assert len(arr) == len(items), f"{name} 임베딩 수({len(arr)}) ≠ items 수({len(items)})"

    # 3) target 라벨 준비
    print("\ntarget 라벨:")
    targets = _prepare_targets(items, prereg)

    # 4) 각 (embedding, target) 조합 probing
    cv = StratifiedKFold(**cv_params)
    results = {}
    for t_name, t_info in targets.items():
        print(f"\n[target: {t_name}] n={t_info['n']:,} classes={t_info['n_classes']}")
        results[t_name] = {}
        y = t_info["y"]
        mask = t_info["mask"]
        for e_name, X_full in emb.items():
            X = X_full[mask]
            print(f"  {e_name:>16} ({X.shape[1]:>3}d) ... ", end="", flush=True)
            r = _probe_one(X, y, cv, clf_params)
            boot_acc = _bootstrap_acc(r["y"], r["y_pred"], n_boot, boot_seed)
            results[t_name][e_name] = {
                "accuracy": r["accuracy"],
                "macro_f1": r["macro_f1"],
                "per_class_f1": r["per_class_f1"],
                "classes": t_info["classes"] if not isinstance(t_info["classes"][0], (int, np.integer)) else [str(c) for c in t_info["classes"]],
                "ci_low": boot_acc["ci_low"], "ci_high": boot_acc["ci_high"],
                "y_pred": r["y_pred"].tolist(),
            }
            print(f"acc={r['accuracy']:.4f} [{boot_acc['ci_low']:.4f}, {boot_acc['ci_high']:.4f}]  macro_f1={r['macro_f1']:.4f}")

    # 5) 확증 검정
    def diff_ci(target, emb_a, emb_b):
        y = targets[target]["y"]
        pa = np.array(results[target][emb_a]["y_pred"])
        pb = np.array(results[target][emb_b]["y_pred"])
        d = _bootstrap_diff(y, pa, pb, n_boot, boot_seed)
        d["supported"] = bool(d["ci_low"] > 0)
        return d

    primary = diff_ci("appearance", "combined", "text")
    primary["comparison"] = "accuracy(combined, appearance) − accuracy(text, appearance)"

    secondary = []
    s1 = diff_ci("appearance", "image", "text")
    s1["comparison"] = "accuracy(image, appearance) − accuracy(text, appearance)"
    secondary.append(s1)

    s2 = diff_ci("desc_mentions_pattern", "text", "image")
    s2["comparison"] = "accuracy(text, desc_mentions_pattern) − accuracy(image, desc_mentions_pattern)"
    secondary.append(s2)

    # shuffled_image chance check
    s3 = _bootstrap_acc(targets["appearance"]["y"],
                        np.array(results["appearance"]["shuffled_image"]["y_pred"]),
                        n_boot, boot_seed)
    chance = 1.0 / targets["appearance"]["n_classes"]
    s3["comparison"] = f"accuracy(shuffled_image, appearance) vs chance ({chance:.4f})"
    s3["chance_level"] = chance
    s3["supported"] = bool(s3["ci_high"] - chance < 0.05)   # chance에서 5%p 이내
    secondary.append(s3)

    # 6) 탐색적 쌍대
    from itertools import combinations
    pairwise = {}
    for t_name in targets:
        pairwise[t_name] = {}
        for a, b in combinations(emb.keys(), 2):
            pairwise[t_name][f"{a}_vs_{b}"] = diff_ci(t_name, a, b)

    # 7) 저장 (y_pred 제거)
    results_out = {}
    for t_name, by_emb in results.items():
        results_out[t_name] = {}
        for e_name, d in by_emb.items():
            d_out = {k: v for k, v in d.items() if k != "y_pred"}
            results_out[t_name][e_name] = d_out

    output = {
        "kind": "D1_representation_probing",
        "created_by": "paper/scripts/analysis_d1_probing.py",
        "prereg": args.prereg,
        "n_boot": n_boot,
        "boot_seed": boot_seed,
        "target_summary": {
            t_name: {"n": t_info["n"], "n_classes": t_info["n_classes"]}
            for t_name, t_info in targets.items()
        },
        "by_target": results_out,
        "confirmatory_primary": primary,
        "confirmatory_secondary": secondary,
        "exploratory_pairwise": pairwise,
    }
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(output, indent=2, ensure_ascii=False, default=float), encoding="utf-8")

    # 8) 요약 출력
    print("\n=== 요약 ===")
    print(f"\n[probing accuracy, 네 target × 네 임베딩]")
    header = f"{'target':<24}" + "".join(f"{e:>14}" for e in emb.keys())
    print(header)
    for t_name in targets:
        row = f"{t_name:<24}"
        for e_name in emb.keys():
            d = results[t_name][e_name]
            row += f"  {d['accuracy']:>12.4f}"
        print(row)

    print(f"\n[주 검정] {primary['comparison']}")
    print(f"  = {primary['estimate']:+.4f} [{primary['ci_low']:+.4f}, {primary['ci_high']:+.4f}]  지지={primary['supported']}")

    print(f"\n[보조 검정]")
    for s in secondary:
        tag = "chance 이내" if "chance_level" in s else "유의"
        print(f"  {s['comparison']}")
        if "chance_level" in s:
            print(f"    chance={s['chance_level']:.4f}, acc={s['estimate']:+.4f} [{s['ci_low']:+.4f}, {s['ci_high']:+.4f}]  지지={s['supported']}")
        else:
            print(f"    = {s['estimate']:+.4f} [{s['ci_low']:+.4f}, {s['ci_high']:+.4f}]  지지={s['supported']}")

    print(f"\n저장: {args.out}")


if __name__ == "__main__":
    main()
