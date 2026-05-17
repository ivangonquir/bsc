"""
FACTORIAL STUDY of the attention-feature ablation.

The 64 combos per detector form a near-complete 2^6 factorial over 6 binary
factors (cls_token_pca + 5 attention blocks); only the empty cell is missing,
and cls_token_raw (1 combo, never combined) is excluded as it carries no
factorial information.

Produces, per detector:
  - main-effects regression  (each feature's average effect on AUROC, with CI)
  - two-way interaction model (which feature pairs synergize / interfere)
  - pooled cross-dataset model (do feature effects differ across datasets?)

All p-values are FDR-corrected (Benjamini-Hochberg) across the full family.
Robust (HC3) standard errors throughout.
"""

import numpy as np
import pandas as pd
import statsmodels.formula.api as smf
import statsmodels.api as sm
from statsmodels.stats.multitest import multipletests

DATASETS  = ["sms_spam", "email_spam", "bbc"]
DATA_TYPE = {"sms_spam": "lexical", "email_spam": "lexical", "bbc": "semantic"}
DETECTORS = ["LOF", "DeepSVDD", "ECOD", "IForest", "SO-GAAL", "AE", "VAE", "LUNAR"]
FEATURES  = ["has_cls_token_pca", "has_cls_mean", "has_cls_max",
             "has_cls_std", "has_head_entropy", "has_diag_mean"]

# ---------------------------------------------------------------- load
frames = []
for ds in DATASETS:
    d = pd.read_csv(f"{ds}_attn_ablation_all_detectors.csv")
    for c in ["has_cls_token_raw", "has_cls_token_pca"]:
        d[c] = d[c].fillna(0).astype(int)
    d["dataset"] = ds
    d["dataset_type"] = DATA_TYPE[ds]
    frames.append(d)
df = pd.concat(frames, ignore_index=True)

# Drop cls_token_raw rows: that feature never varies in combination,
# so it cannot be a factor in the factorial.
df = df[df["has_cls_token_raw"] == 0].copy()
print(f"Factorial design: {df.groupby(['dataset','detector']).size().iloc[0]} "
      f"combos per (dataset, detector)")
print(f"Total rows: {len(df)}  ({len(DATASETS)} datasets x {len(DETECTORS)} detectors)\n")


# ================================================================
# PART 1 — Main-effects regression per (dataset, detector)
# ================================================================
main_rows  = []
fit_rows   = []
formula_main = "auroc_mean ~ " + " + ".join(FEATURES)

for ds in DATASETS:
    for det in DETECTORS:
        sub = df[(df["dataset"] == ds) & (df["detector"] == det)]
        model = smf.ols(formula_main, data=sub).fit(cov_type="HC3")
        fit_rows.append({
            "dataset": ds, "detector": det,
            "r_squared": model.rsquared,
            "adj_r_squared": model.rsquared_adj,
            "n_obs": int(model.nobs),
        })
        ci = model.conf_int(alpha=0.05)
        for feat in FEATURES:
            main_rows.append({
                "dataset":   ds,
                "detector":  det,
                "feature":   feat.replace("has_", ""),
                "coef":      model.params[feat],
                "std_err":   model.bse[feat],
                "ci95_low":  ci.loc[feat, 0],
                "ci95_high": ci.loc[feat, 1],
                "t_stat":    model.tvalues[feat],
                "p_value":   model.pvalues[feat],
            })

main_df = pd.DataFrame(main_rows)
fit_df  = pd.DataFrame(fit_rows)

# FDR-correct across the whole family of main-effect tests
main_df["p_value_fdr"] = multipletests(main_df["p_value"], method="fdr_bh")[1]
main_df["sig"] = np.where(main_df["p_value_fdr"] >= 0.05, "",
                  np.where(main_df["coef"] > 0, "+", "-"))
main_df = main_df.round(4)
main_df.to_csv("factorial_main_effects.csv", index=False)
fit_df.round(4).to_csv("factorial_model_fit.csv", index=False)

print("=" * 78)
print("PART 1 — MAIN EFFECTS  (coef = avg AUROC change from adding the feature)")
print("=" * 78)
print(f"Main-effects model R^2: min={fit_df.r_squared.min():.3f}  "
      f"mean={fit_df.r_squared.mean():.3f}  max={fit_df.r_squared.max():.3f}")
print("(high R^2 => main effects dominate; low => interactions matter)\n")


# ================================================================
# PART 2 — Feature regimes: consistency across the 24 models
# ================================================================
print("=" * 78)
print("PART 2 — FEATURE REGIMES  (significant effects across 24 detector x dataset cells)")
print("=" * 78)
regime_rows = []
for feat in [f.replace("has_", "") for f in FEATURES]:
    fsub = main_df[main_df["feature"] == feat]
    n_pos = (fsub["sig"] == "+").sum()
    n_neg = (fsub["sig"] == "-").sum()
    n_ns  = (fsub["sig"] == "").sum()
    regime_rows.append({
        "feature": feat,
        "sig_positive": n_pos,
        "sig_negative": n_neg,
        "not_sig": n_ns,
        "mean_coef": fsub["coef"].mean(),
        "coef_range": f"[{fsub['coef'].min():.3f}, {fsub['coef'].max():.3f}]",
    })
regime_df = pd.DataFrame(regime_rows).sort_values("mean_coef", ascending=False)
regime_df.to_csv("factorial_feature_regimes.csv", index=False)
print(regime_df.round(4).to_string(index=False))
print()


# ================================================================
# PART 3 — Two-way interactions per (dataset, detector)
# ================================================================
inter_rows = []
formula_int = ("auroc_mean ~ (" + " + ".join(FEATURES) + ")**2")

for ds in DATASETS:
    for det in DETECTORS:
        sub = df[(df["dataset"] == ds) & (df["detector"] == det)]
        model = smf.ols(formula_int, data=sub).fit(cov_type="HC3")
        for term in model.params.index:
            if ":" not in term:
                continue
            a, b = term.split(":")
            inter_rows.append({
                "dataset":  ds,
                "detector": det,
                "feat_A":   a.replace("has_", ""),
                "feat_B":   b.replace("has_", ""),
                "coef":     model.params[term],
                "p_value":  model.pvalues[term],
            })

inter_df = pd.DataFrame(inter_rows)
inter_df["p_value_fdr"] = multipletests(inter_df["p_value"], method="fdr_bh")[1]
inter_df["sig"] = np.where(inter_df["p_value_fdr"] >= 0.05, "",
                   np.where(inter_df["coef"] > 0, "synergy", "interference"))
inter_df = inter_df.round(4)
inter_df.to_csv("factorial_interactions.csv", index=False)

sig_inter = inter_df[inter_df["sig"] != ""]
print("=" * 78)
print("PART 3 — TWO-WAY INTERACTIONS")
print("=" * 78)
print(f"{len(sig_inter)} of {len(inter_df)} two-way interaction terms significant "
      f"(FDR<0.05)")
print("  synergy      = pair does better than the sum of its parts")
print("  interference = pair does worse (redundant or conflicting)\n")
if not sig_inter.empty:
    summ = sig_inter.groupby(["feat_A", "feat_B", "sig"]).size().reset_index(name="n_cells")
    print("Most consistent interactions (count over 24 detector x dataset cells):")
    print(summ.sort_values("n_cells", ascending=False).head(12).to_string(index=False))
print()


# ================================================================
# PART 4 — Cross-dataset heterogeneity per detector
#  Does each detector's set of feature effects differ across datasets?
# ================================================================
print("=" * 78)
print("PART 4 — CROSS-DATASET HETEROGENEITY  (feature x dataset interaction)")
print("=" * 78)
print("H0: feature effects are the same across all 3 datasets.")
print("Small p => feature effects are dataset-dependent.\n")

het_rows = []
reduced_f = "auroc_mean ~ " + " + ".join(FEATURES) + " + C(dataset)"
full_f    = ("auroc_mean ~ (" + " + ".join(FEATURES) + ") * C(dataset)")
for det in DETECTORS:
    sub = df[df["detector"] == det]
    m_red  = smf.ols(reduced_f, data=sub).fit()
    m_full = smf.ols(full_f,    data=sub).fit()
    anova = sm.stats.anova_lm(m_red, m_full)
    F  = anova["F"].iloc[1]
    p  = anova["Pr(>F)"].iloc[1]
    het_rows.append({
        "detector": det,
        "F_stat": F,
        "p_value": p,
        "main_only_R2": m_red.rsquared,
        "with_interaction_R2": m_full.rsquared,
    })
het_df = pd.DataFrame(het_rows)
het_df["p_value_fdr"] = multipletests(het_df["p_value"], method="fdr_bh")[1]
het_df["dataset_dependent"] = np.where(het_df["p_value_fdr"] < 0.05, "YES", "no")
het_df = het_df.round(4)
het_df.to_csv("factorial_dataset_heterogeneity.csv", index=False)
print(het_df.to_string(index=False))
print()

print("=" * 78)
print("FILES WRITTEN")
print("=" * 78)
for f in ["factorial_main_effects.csv", "factorial_model_fit.csv",
          "factorial_feature_regimes.csv", "factorial_interactions.csv",
          "factorial_dataset_heterogeneity.csv"]:
    print(f"  {f}")
