"""
Attention-feature ablation on sms_spam — entry point.

Runs ablation over all 31+ feature combinations × available detectors,
saves per-(combo, detector) results, and reports marginal feature importance.
GPU-heavy detectors are skipped automatically when no CUDA device is found.
"""

from config import RESULTS_CSV, IMPORTANCE_CSV, DETECTOR_NAMES
from ablation import run_ablation, feature_importance


if __name__ == "__main__":
    df = run_ablation()
    df.to_csv(RESULTS_CSV, index=False)
    print(f"\nFinal results saved -> {RESULTS_CSV}")

    print("\n=== Top 10 (combo, detector) by AUROC ===")
    print(df.sort_values("auroc_mean", ascending=False).head(10)[
        ["combo", "detector", "dim", "auroc_mean", "auroc_std"]
    ].to_string(index=False))

    print("\n=== Singletons by detector ===")
    print(df[df["n_blocks"] == 1].sort_values(
        ["detector", "auroc_mean"], ascending=[True, False])[
        ["detector", "combo", "auroc_mean", "auroc_std"]
    ].to_string(index=False))

    imp = feature_importance(df)
    imp.to_csv(IMPORTANCE_CSV, index=False)
    print(f"\n=== Marginal feature importance per detector ===")
    print(imp.to_string(index=False))

    print("\n=== Importance pivot (rows=feature, cols=detector, values=delta) ===")
    pivot = imp.pivot(index="feature", columns="detector", values="delta")
    print(pivot.round(4).to_string())

    print("\n=== Robust ranking (mean delta across detectors) ===")
    det_set   = set(DETECTOR_NAMES)
    sub       = imp[imp["detector"].isin(det_set)]
    mean_d    = sub.groupby("feature")["delta"].mean().sort_values(ascending=False)
    agree_d   = sub.groupby("feature")["delta"].apply(lambda s: int((s > 0).sum()))
    mean_dict  = mean_d.to_dict()
    agree_dict = agree_d.to_dict()
    for feat in mean_d.index:
        print(f"  {feat:<22}  mean_delta={mean_dict[feat]:+.4f}  "
              f"agree={agree_dict[feat]}/{len(det_set)}")
