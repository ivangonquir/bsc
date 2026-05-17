"""
Cross-dataset analysis: sms_spam, email_spam, bbc.
Tests whether the patterns from sms_spam replicate.
"""
import pandas as pd
import numpy as np

PAPER_BERT = {
    "sms_spam":   {"LOF":0.7190,"DeepSVDD":0.5859,"ECOD":0.5606,"IForest":0.5053,
                   "SO-GAAL":0.3328,"AE":0.6918,"VAE":0.6082,"LUNAR":0.6953},
    "email_spam": {"LOF":0.7482,"DeepSVDD":0.6937,"ECOD":0.7052,"IForest":0.6779,
                   "SO-GAAL":0.4440,"AE":0.4739,"VAE":0.4737,"LUNAR":0.8417},
    "bbc":        {"LOF":0.9320,"DeepSVDD":0.5683,"ECOD":0.6912,"IForest":0.6847,
                   "SO-GAAL":0.3099,"AE":0.8839,"VAE":0.7409,"LUNAR":0.9260},
}
PAPER_OPENAI = {
    "sms_spam":   {"LOF":0.7862,"DeepSVDD":0.3491,"ECOD":0.4317,"IForest":0.3751,
                   "SO-GAAL":0.5671,"AE":0.5511,"VAE":0.4259,"LUNAR":0.7189},
    "email_spam": {"LOF":0.9263,"DeepSVDD":0.4415,"ECOD":0.9263,"IForest":0.6937,
                   "SO-GAAL":0.4440,"AE":0.7651,"VAE":0.5273,"LUNAR":0.9343},
    "bbc":        {"LOF":0.9558,"DeepSVDD":0.5766,"ECOD":0.7224,"IForest":0.6064,
                   "SO-GAAL":0.2359,"AE":0.9520,"VAE":0.7250,"LUNAR":0.9732},
}

DETECTORS = ["LOF","DeepSVDD","ECOD","IForest","SO-GAAL","AE","VAE","LUNAR"]
DATASETS  = ["sms_spam", "email_spam", "bbc"]
DATASET_DIRS = {
    "sms_spam":   "sms-spam",
    "email_spam": "email_spam-out",
    "bbc":        "bbc-out",
}

dfs = {}
for ds in DATASETS:
    path = f"{DATASET_DIRS[ds]}/{ds}_attn_ablation_all_detectors.csv"
    d = pd.read_csv(path)
    d["has_cls_token_raw"] = d["has_cls_token_raw"].fillna(0).astype(int)
    d["has_cls_token_pca"] = d["has_cls_token_pca"].fillna(0).astype(int)
    dfs[ds] = d

# ---- 1. Per-detector best, all three datasets ----
print("="*100)
print("BEST OVERALL per detector vs paper baselines")
print("="*100)
summary = []
for ds in DATASETS:
    d = dfs[ds]
    for det in DETECTORS:
        sub = d[d["detector"] == det]
        best = sub.loc[sub["auroc_mean"].idxmax()]
        cls_raw = sub[sub["combo"]=="cls_token_raw"]["auroc_mean"]
        cls_raw = cls_raw.iloc[0] if not cls_raw.empty else np.nan
        summary.append({
            "dataset": ds, "detector": det,
            "paper_BERT": PAPER_BERT[ds][det],
            "paper_OpenAI": PAPER_OPENAI[ds][det],
            "our_cls_raw": round(cls_raw,4),
            "best_overall": round(best["auroc_mean"],4),
            "best_combo": best["combo"],
            "gain_vs_BERT": round(best["auroc_mean"]-PAPER_BERT[ds][det],4),
        })
sdf = pd.DataFrame(summary)
sdf.to_csv("cross_dataset_summary.csv", index=False)
for ds in DATASETS:
    print(f"\n--- {ds} ---")
    print(sdf[sdf["dataset"]==ds].drop(columns="dataset").to_string(index=False))

# ---- 2. cls_mean + IForest anti-prediction check ----
print("\n" + "="*100)
print("cls_mean + IForest anti-prediction: does it replicate?")
print("="*100)
for ds in DATASETS:
    d = dfs[ds]
    row = d[(d["combo"]=="cls_mean") & (d["detector"]=="IForest")]
    if not row.empty:
        r = row.iloc[0]
        print(f"  {ds:12s}: cls_mean+IForest AUROC = {r['auroc_mean']:.4f} ± {r['auroc_std']:.4f}")

# ---- 3. head_entropy for density detectors (LOF, LUNAR) ----
print("\n" + "="*100)
print("head_entropy singleton for LOF / LUNAR: does the density-detector affinity hold?")
print("="*100)
for ds in DATASETS:
    d = dfs[ds]
    for det in ["LOF","LUNAR"]:
        row = d[(d["combo"]=="head_entropy") & (d["detector"]==det)]
        if not row.empty:
            r = row.iloc[0]
            print(f"  {ds:12s} {det:7s}: head_entropy alone AUROC = {r['auroc_mean']:.4f}")

# ---- 4. SO-GAAL on attention-only ----
print("\n" + "="*100)
print("SO-GAAL best attention-only combo per dataset")
print("="*100)
for ds in DATASETS:
    d = dfs[ds]
    sub = d[(d["detector"]=="SO-GAAL") &
            (d["has_cls_token_raw"]==0) & (d["has_cls_token_pca"]==0)]
    if not sub.empty:
        best = sub.loc[sub["auroc_mean"].idxmax()]
        print(f"  {ds:12s}: {best['combo']:45s} AUROC = {best['auroc_mean']:.4f}  "
              f"(paper best BERT={max(PAPER_BERT[ds].values()):.4f})")

# ---- 5. Count: how often does our best beat paper BERT / OpenAI ----
print("\n" + "="*100)
print("Win count: best_overall vs paper baselines")
print("="*100)
for ds in DATASETS:
    d = dfs[ds]
    wins_bert = wins_openai = 0
    for det in DETECTORS:
        sub = d[d["detector"]==det]
        best = sub["auroc_mean"].max()
        if best > PAPER_BERT[ds][det]:   wins_bert += 1
        if best > PAPER_OPENAI[ds][det]: wins_openai += 1
    print(f"  {ds:12s}: beats paper BERT in {wins_bert}/8 detectors, "
          f"beats paper OpenAI in {wins_openai}/8")

print("\nSaved -> cross_dataset_summary.csv")