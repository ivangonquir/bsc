"""
Attention-feature ablation study — entry point.

Runs ablation over all 31+ feature combinations × available detectors,
saves per-(combo, detector) results, and reports marginal feature importance.
GPU-heavy detectors are skipped automatically when no CUDA device is found.

Usage:
    python main.py                        # defaults to sms_spam
    python main.py --dataset email_spam
    python main.py --dataset bbc
"""

import argparse
import pandas as pd
from config import RunConfig, DATASET_CONFIG, DETECTOR_NAMES
from ablation import run_ablation, feature_importance


def parse_args():
    parser = argparse.ArgumentParser(description="Attention-feature ablation study")
    parser.add_argument(
        "--dataset",
        choices=list(DATASET_CONFIG),
        default="sms_spam",
        help="Dataset to run (default: sms_spam)",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args    = parse_args()
    run_cfg = RunConfig.from_dataset(args.dataset)

    print(f"Dataset : {run_cfg.subdataset}  (max_len={run_cfg.max_len})")
    print(f"Results : {run_cfg.results_csv}")

    df = run_ablation(run_cfg)
    df.to_csv(run_cfg.results_csv, index=False)
    print(f"\nFinal results saved -> {run_cfg.results_csv}")

    print("\n=== Top 10 (combo, detector) by AUROC ===")
    top10 = df.sort_values("auroc_mean").iloc[::-1].head(10)
    print(top10[["combo", "detector", "dim", "auroc_mean", "auroc_std"]].to_string(index=False))

    print("\n=== Singletons by detector ===")
    sngl = df.loc[df["n_blocks"] == 1][["detector", "combo", "auroc_mean", "auroc_std"]]
    sngl_rows = []
    for det in sorted(sngl["detector"].unique()):
        grp = sngl.loc[sngl["detector"] == det].sort_values("auroc_mean").iloc[::-1]
        sngl_rows.append(grp)
    print(pd.concat(sngl_rows).to_string(index=False))

    imp = feature_importance(df)
    imp.to_csv(run_cfg.importance_csv, index=False)
    print(f"\n=== Marginal feature importance per detector ===")
    print(imp.to_string(index=False))

    print("\n=== Importance pivot (rows=feature, cols=detector, values=delta) ===")
    pivot = imp.pivot(index="feature", columns="detector", values="delta")
    print(pivot.round(4).to_string())

    print("\n=== Robust ranking (mean delta across detectors) ===")
    det_set    = set(DETECTOR_NAMES)
    sub        = imp[imp["detector"].isin(det_set)]
    mean_d     = sub.groupby("feature")["delta"].mean().sort_values().iloc[::-1]
    agree_d    = sub.groupby("feature")["delta"].apply(lambda s: int((s > 0).sum()))
    mean_dict  = mean_d.to_dict()
    agree_dict = agree_d.to_dict()
    for feat in mean_d.index:
        print(f"  {feat:<22}  mean_delta={mean_dict[feat]:+.4f}  "
              f"agree={agree_dict[feat]}/{len(det_set)}")
