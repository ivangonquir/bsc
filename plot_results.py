"""
Generate figures for the README.

Figure 1 — feature_importance.png: marginal ΔAUROC per (feature, detector), SMS Spam.
Figure 2 — gain_vs_bert.png: best attention combo vs. paper BERT baseline across datasets.

Run from the project root:
    python plot_results.py
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

FIGURES_DIR = "figures"
os.makedirs(FIGURES_DIR, exist_ok=True)

DATASETS  = ["sms_spam", "email_spam", "bbc"]
DETECTORS = ["LOF", "DeepSVDD", "ECOD", "IForest", "SO-GAAL", "AE", "VAE", "LUNAR"]
DATASET_DIRS = {
    "sms_spam":   "sms-spam",
    "email_spam": "email_spam-out",
    "bbc":        "bbc-out",
}
DATASET_LABELS = {
    "sms_spam":   "SMS Spam",
    "email_spam": "Email Spam",
    "bbc":        "BBC News",
}
FEATURE_ORDER = [
    "cls_token_raw", "cls_token_pca",
    "cls_mean", "cls_max", "cls_std", "head_entropy", "diag_mean",
]
PAPER_BERT = {
    "sms_spam":   {"LOF":0.7190,"DeepSVDD":0.5859,"ECOD":0.5606,"IForest":0.5053,
                   "SO-GAAL":0.3328,"AE":0.6918,"VAE":0.6082,"LUNAR":0.6953},
    "email_spam": {"LOF":0.7482,"DeepSVDD":0.6937,"ECOD":0.7052,"IForest":0.6779,
                   "SO-GAAL":0.4440,"AE":0.4739,"VAE":0.4737,"LUNAR":0.8417},
    "bbc":        {"LOF":0.9320,"DeepSVDD":0.5683,"ECOD":0.6912,"IForest":0.6847,
                   "SO-GAAL":0.3099,"AE":0.8839,"VAE":0.7409,"LUNAR":0.9260},
}


def load_ablation_csvs() -> dict[str, pd.DataFrame]:
    dfs = {}
    for ds in DATASETS:
        path = os.path.join(DATASET_DIRS[ds], f"{ds}_attn_ablation_all_detectors.csv")
        d = pd.read_csv(path)
        d["has_cls_token_raw"] = d["has_cls_token_raw"].fillna(0).astype(int)
        d["has_cls_token_pca"] = d["has_cls_token_pca"].fillna(0).astype(int)
        dfs[ds] = d
    return dfs


def plot_feature_importance(importance_csv: str, out_path: str) -> None:
    """Heatmap of marginal ΔAUROC per (feature, detector) for SMS Spam."""
    imp = pd.read_csv(importance_csv)
    pivot = imp.pivot(index="feature", columns="detector", values="delta")

    det_order  = [d for d in DETECTORS     if d in pivot.columns]
    feat_order = [f for f in FEATURE_ORDER if f in pivot.index]
    pivot = pivot.loc[feat_order, det_order]

    fig, ax = plt.subplots(figsize=(11, 4.5))
    sns.heatmap(
        pivot, ax=ax,
        annot=True, fmt=".3f", linewidths=0.5,
        cmap="RdBu_r", center=0,
        cbar_kws={"label": "ΔAUROC (with − without feature)", "shrink": 0.8},
    )
    ax.set_title("Marginal Feature Importance — SMS Spam", fontsize=13, pad=12)
    ax.set_xlabel("Detector")
    ax.set_ylabel("Feature")
    plt.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {out_path}")


def plot_gain_vs_bert(dfs: dict[str, pd.DataFrame], out_path: str) -> None:
    """Heatmap of (best attention AUROC − paper BERT AUROC) across datasets × detectors."""
    gain_rows = {}
    for ds in DATASETS:
        d = dfs[ds]
        attn_only = d[d["has_cls_token_raw"] == 0]
        gains = {}
        for det in DETECTORS:
            sub = attn_only[attn_only["detector"] == det]
            gains[det] = (
                round(float(sub["auroc_mean"].max()) - PAPER_BERT[ds][det], 4)
                if not sub.empty else np.nan
            )
        gain_rows[DATASET_LABELS[ds]] = gains

    gain_df = pd.DataFrame(gain_rows, index=DETECTORS).T

    fig, ax = plt.subplots(figsize=(11, 3))
    sns.heatmap(
        gain_df, ax=ax,
        annot=True, fmt=".3f", linewidths=0.5,
        cmap="RdBu_r", center=0,
        cbar_kws={"label": "ΔAUROC vs. BERT baseline", "shrink": 0.8},
    )
    ax.set_title("Best Attention Combo − Paper BERT Baseline", fontsize=13, pad=12)
    ax.set_xlabel("Detector")
    ax.set_ylabel("Dataset")
    ax.set_yticklabels(ax.get_yticklabels(), rotation=0)
    plt.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {out_path}")


def plot_feature_diagram(out_path: str) -> None:
    """Schematic showing which part of the attention matrix each feature is computed from."""
    import matplotlib.patches as mpatches

    np.random.seed(0)
    tokens = ["[CLS]", "spam", "click", "here", "now", "[SEP]"]
    n = len(tokens)

    # Synthetic attention weights for one head (softmax over rows).
    logits = np.random.randn(n, n)
    logits[0] = [-0.3, 1.5, 0.9, 0.2, 1.1, -0.4]  # make CLS row visually interesting
    attn = np.exp(logits) / np.exp(logits).sum(axis=1, keepdims=True)

    fig, (ax_mat, ax_leg) = plt.subplots(
        1, 2, figsize=(13, 5.5), gridspec_kw={"width_ratios": [1, 0.75]}
    )

    # ── Left: attention matrix with overlays ──────────────────────────────────
    # Grey base so coloured overlays are clearly distinguishable.
    im = ax_mat.imshow(attn, cmap="Greys", aspect="equal", vmin=0)
    ax_mat.set_xticks(range(n))
    ax_mat.set_yticks(range(n))
    ax_mat.set_xticklabels(tokens, fontsize=10)
    ax_mat.set_yticklabels(tokens, fontsize=10)
    ax_mat.set_xlabel("Key  (attended-to token)", fontsize=10)
    ax_mat.set_ylabel("Query  (attending token)", fontsize=10)
    ax_mat.set_title("Attention matrix — one head, last BERT layer", fontsize=11, pad=10)
    plt.colorbar(im, ax=ax_mat, shrink=0.8, label="Attention weight")

    CLS_COLOR  = "#e74c3c"
    DIAG_COLOR = "#2ecc71"
    ENT_COLOR  = "#9b59b6"

    # Red overlay on CLS row (row 0) → cls_mean, cls_max, cls_std
    ax_mat.add_patch(mpatches.Rectangle(
        (-0.5, -0.5), n, 1, lw=3, ec="#c0392b", fc=CLS_COLOR, alpha=0.55, zorder=2
    ))

    # Green overlay on diagonal → diag_mean
    for i in range(n):
        ax_mat.add_patch(mpatches.Rectangle(
            (i - 0.5, i - 0.5), 1, 1, lw=2.5, ec="#27ae60", fc=DIAG_COLOR, alpha=0.65, zorder=2
        ))

    # Purple border around full matrix → head_entropy (spans all rows/heads)
    ax_mat.add_patch(mpatches.Rectangle(
        (-0.5, -0.5), n, n, lw=4, ec=ENT_COLOR, fc="none", zorder=3
    ))

    # ── Right: feature legend ─────────────────────────────────────────────────
    ax_leg.axis("off")
    ax_leg.set_title("Extracted features", fontsize=11, pad=10)

    groups = [
        (CLS_COLOR,  "CLS row  (red overlay)", [
            ("cls_mean",     "Mean attention weight from [CLS] to each token"),
            ("cls_max",      "Max attention weight from [CLS] to any token"),
            ("cls_std",      "Std deviation of [CLS] attention weights"),
        ]),
        (DIAG_COLOR, "Diagonal  (green overlay)", [
            ("diag_mean",    "Mean of self-attention diagonal elements"),
        ]),
        (ENT_COLOR,  "Full matrix  (purple border)", [
            ("head_entropy", "Shannon entropy of each head's row distributions,\n"
                             "averaged over all heads and non-padding tokens"),
        ]),
    ]

    y = 0.97
    for color, label, features in groups:
        ax_leg.text(0.0, y, label, fontsize=10, fontweight="bold", color=color,
                    va="top", transform=ax_leg.transAxes)
        y -= 0.07
        for name, desc in features:
            ax_leg.text(0.04, y, f"• {name}", fontsize=9.5, fontweight="bold",
                        color="#2c3e50", va="top", transform=ax_leg.transAxes)
            y -= 0.055
            ax_leg.text(0.06, y, desc, fontsize=8.5, color="#555555",
                        va="top", transform=ax_leg.transAxes)
            y -= 0.075
        y -= 0.03

    plt.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {out_path}")


if __name__ == "__main__":
    dfs = load_ablation_csvs()

    plot_feature_diagram(
        out_path=os.path.join(FIGURES_DIR, "feature_diagram.png"),
    )
    plot_feature_importance(
        importance_csv=os.path.join(DATASET_DIRS["sms_spam"],
                                    "sms_spam_attn_importance_all_detectors.csv"),
        out_path=os.path.join(FIGURES_DIR, "feature_importance.png"),
    )
    plot_gain_vs_bert(
        dfs=dfs,
        out_path=os.path.join(FIGURES_DIR, "gain_vs_bert.png"),
    )
