"""
Attention-feature ablation on sms_spam, full dataset, all 8 detectors, 5 seeds.

5 attention blocks (cls_mean, cls_max, cls_std, head_entropy, diag_mean),
31 non-empty combinations, 8 detectors:
  Stochastic (5 seeds): IForest, DeepSVDD, AE, VAE
  Deterministic (1 run): LOF, ECOD, LUNAR, SO-GAAL

Resumable: skips (combo, detector) pairs already in the CSV. Partial CSV
written after every (combo, detector) so crashes don't lose work.
"""

""" 
Applying StandardScaler to Transformer embeddings is generally a critical error. 
BERT encodes semantic meaning in the relative magnitudes and angles across its 768 dimensions. StandardScaler normalizes each 
of the 768 dimensions independently across the entire dataset.

By forcing every single dimension to have a mean of 0 and a variance of 1, you arbitrarily stretch and squash the 
hypersphere geometry of the embeddings. A dimension that originally had very low variance (perhaps representing a rare, 
highly specific semantic concept) is blown up to have the exact same mathematical weight as a primary structural dimension. 
Distance-based detectors like LOF, ECOD, and DeepSVDD rely entirely on the original spatial geometry to find outliers. 
By squashing that space, you obscure the very anomaly signals these models are looking for.

Furthermore, since you are specifically researching VAEs for anomaly detection, you know that altering 
the variance of the input space directly impacts how the encoder maps data to the Gaussian latent space; standardizing
the input often forces generative models to waste capacity reconstructing scaled noise rather than semantic structure.

The Fix: The authors likely passed the embeddings into the models raw, or they applied sample-wise L2 Normalization
(making each sentence vector length 1). Comment out the StandardScaler lines entirely when running your cls_token baseline.
"""

import os
import gc
import time
from itertools import combinations

import numpy as np
import pandas as pd
import torch

import transformers
transformers.logging.set_verbosity_error()
from transformers import AutoTokenizer, AutoModel
from datasets import load_dataset

from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler, normalize
from sklearn.metrics import roc_auc_score, average_precision_score


from pyod.models.lof          import LOF
from pyod.models.deep_svdd    import DeepSVDD
from pyod.models.ecod         import ECOD
from pyod.models.iforest      import IForest
from pyod.models.so_gaal      import SO_GAAL
from pyod.models.auto_encoder import AutoEncoder
from pyod.models.vae          import VAE
from pyod.models.lunar        import LUNAR




# -----------------------------------------------------------------------------
# Config
# -----------------------------------------------------------------------------
HF_REPO        = "kendx/NLP-ADBench"
MODEL_NAME     = "bert-base-uncased"
SUBDATASET     = "sms_spam"
MAX_LEN        = 32
BATCH_SIZE     = 16
DEVICE         = "cuda" if torch.cuda.is_available() else "cpu"

ATTENTION_BLOCKS = ["cls_mean", "cls_max", "cls_std", "head_entropy", "diag_mean"]

# Detectors that rely on neural-network training and are impractically slow on CPU.
# They are skipped automatically when no CUDA GPU is available.
GPU_HEAVY_DETECTORS = {"DeepSVDD", "AE", "VAE", "LUNAR", "SO-GAAL"}
_ALL_DETECTOR_NAMES = ["LOF", "DeepSVDD", "ECOD", "IForest", "SO-GAAL", "AE", "VAE", "LUNAR"]

if DEVICE == "cpu":
    DETECTOR_NAMES = [d for d in _ALL_DETECTOR_NAMES if d not in GPU_HEAVY_DETECTORS]
    print(f"[WARNING] No CUDA GPU detected — skipping GPU-heavy detectors: "
          f"{sorted(GPU_HEAVY_DETECTORS)}.\n"
          f"          Running CPU-only detectors: {DETECTOR_NAMES}")
else:
    DETECTOR_NAMES = _ALL_DETECTOR_NAMES

# Seed configuration: stochastic detectors get 5 seeds, deterministic 1.
DETERMINISTIC = {"LOF", "ECOD", "LUNAR", "SO-GAAL"}
STOCHASTIC_SEEDS = (0, 1, 2)


from huggingface_hub import hf_hub_download

def load_paper_cls_token():
    """Download the paper's precomputed bert-base-uncased CLS embeddings."""
    train_path = hf_hub_download(
        repo_id=HF_REPO, repo_type="dataset",
        filename=f"embeddings/{SUBDATASET}/{SUBDATASET}_train_data_bert_base_uncased_feature.npy",
    )
    test_path = hf_hub_download(
        repo_id=HF_REPO, repo_type="dataset",
        filename=f"embeddings/{SUBDATASET}/{SUBDATASET}_test_data_bert_base_uncased_feature.npy",
    )
    return np.load(train_path), np.load(test_path)


def make_detector(name, seed, input_dim):
    if name == "LOF":       return LOF(n_jobs=-1)
    if name == "ECOD":      return ECOD(n_jobs=-1)
    if name == "LUNAR":     return LUNAR()
    if name == "SO-GAAL":   return SO_GAAL()
    if name == "IForest":   return IForest(random_state=seed, n_jobs=-1)
    if name == "DeepSVDD":  return DeepSVDD(n_features=input_dim,
                                            random_state=seed, verbose=0)
    if name == "AE":        return AutoEncoder(random_state=seed, verbose=0)
    if name == "VAE":       return VAE(random_state=seed, verbose=0)
    raise ValueError(name)


def seeds_for(name):
    return (0,) if name in DETERMINISTIC else STOCHASTIC_SEEDS


FEATURE_FILE = f"{SUBDATASET}_features_{len(ATTENTION_BLOCKS)}blocks_plus_cls_maxlen{MAX_LEN}.npz"
RESULTS_CSV    = f"{SUBDATASET}_attn_ablation_all_detectors.csv"
IMPORTANCE_CSV = f"{SUBDATASET}_attn_importance_all_detectors.csv"


# -----------------------------------------------------------------------------
# Feature loading / extraction (cached)
# -----------------------------------------------------------------------------
def load_split(filename):
    ds = load_dataset(HF_REPO, data_files=filename, split="train")
    return list(ds["text"]), np.array(ds["label"])


@torch.no_grad()
def extract_attention_blocks(texts, tokenizer, model, layer_idx):
    blocks = {b: [] for b in ATTENTION_BLOCKS}
    for start in range(0, len(texts), BATCH_SIZE):
        batch = texts[start:start + BATCH_SIZE]
        enc = tokenizer(batch, padding="max_length", truncation=True,
                        max_length=MAX_LEN, return_tensors="pt").to(DEVICE)
        out  = model(**enc)
        attn = out.attentions[layer_idx]
        mask = enc["attention_mask"].float()
        denom = mask.sum(1).clamp_min(1).unsqueeze(1)

        cls_row = attn[:, :, 0, :] * mask.unsqueeze(1)
        blocks["cls_mean"].append((cls_row.sum(-1) / denom).cpu().numpy())
        blocks["cls_max"].append(cls_row.max(-1).values.cpu().numpy())
        blocks["cls_std"].append(cls_row.std(-1).cpu().numpy())
    

        ent_rows = torch.special.entr(attn).sum(-1)
        blocks["head_entropy"].append(
            ((ent_rows * mask.unsqueeze(1)).sum(-1) / denom).cpu().numpy()
        )
        diag = torch.diagonal(attn, dim1=-2, dim2=-1)
        blocks["diag_mean"].append(
            ((diag * mask.unsqueeze(1)).sum(-1) / denom).cpu().numpy()
        )

        if (start // BATCH_SIZE) % 50 == 0:
            print(f"  ... {min(start + len(batch), len(texts))}/{len(texts)}")
            
    print(blocks.keys())

    return {b: np.concatenate(v, axis=0).astype(np.float32) for b, v in blocks.items()}


def load_or_extract():
    # 1. Fixed Cache Loader to include the two new token baselines
    all_features = ATTENTION_BLOCKS + ["cls_token_raw", "cls_token_pca"]
    
    if os.path.exists(FEATURE_FILE):
        print(f"Loading cached features from {FEATURE_FILE}")
        d = np.load(FEATURE_FILE)
        train_b = {b: d[f"train_{b}"] for b in all_features}
        test_b  = {b: d[f"test_{b}"]  for b in all_features}
        return train_b, test_b, d["test_labels"]

    print("Extracting attention features fresh...")
    train_texts, _      = load_split(f"datasets/{SUBDATASET}/{SUBDATASET}_train_data.jsonl")
    test_texts, y_test  = load_split(f"datasets/{SUBDATASET}/{SUBDATASET}_test_data.jsonl")

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModel.from_pretrained(
        MODEL_NAME, output_attentions=True, attn_implementation="eager",
    ).to(DEVICE).eval()
    layer_idx = model.config.num_hidden_layers - 1
    
    print("Train features:")
    train_b = extract_attention_blocks(train_texts, tokenizer, model, layer_idx)
    print("Test features:")
    test_b  = extract_attention_blocks(test_texts, tokenizer, model, layer_idx)
    
    print("Loading paper's precomputed cls_token embeddings...")
    paper_train_cls, paper_test_cls = load_paper_cls_token()
    
    # Sanity check: row counts must match
    n_train = len(next(iter(train_b.values())))
    n_test  = len(next(iter(test_b.values())))
    assert paper_train_cls.shape[0] == n_train, \
        f"train row mismatch: paper {paper_train_cls.shape[0]} vs ours {n_train}"
    assert paper_test_cls.shape[0] == n_test, \
        f"test row mismatch: paper {paper_test_cls.shape[0]} vs ours {n_test}"

    # Save the absolute raw token for the paper baseline
    train_b["cls_token_raw"] = paper_train_cls.astype(np.float32)
    test_b["cls_token_raw"]  = paper_test_cls.astype(np.float32)
    
    # Pre-compute the PCA version for the fair ablation combinations
    from sklearn.decomposition import PCA
    pca = PCA(n_components=64, random_state=42)
    train_b["cls_token_pca"] = pca.fit_transform(paper_train_cls).astype(np.float32)
    test_b["cls_token_pca"]  = pca.transform(paper_test_cls).astype(np.float32)
    print(f"PCA(64) explained variance: {pca.explained_variance_ratio_.sum():.3f}")

    # 2. Fixed Save logic to happen exactly once
    save = {f"train_{b}": train_b[b] for b in all_features}
    save.update({f"test_{b}": test_b[b] for b in all_features})
    save["test_labels"] = y_test
    
    np.savez_compressed(FEATURE_FILE, **save)
    print(f"Saved features to {FEATURE_FILE}")

    del model, tokenizer
    if torch.cuda.is_available(): torch.cuda.empty_cache()
    gc.collect()
    
    return train_b, test_b, y_test


# -----------------------------------------------------------------------------
# Resumability: load existing CSV and figure out what's already done
# -----------------------------------------------------------------------------
def load_existing_results():
    if not os.path.exists(RESULTS_CSV):
        return pd.DataFrame(), set()
    df = pd.read_csv(RESULTS_CSV)
    done = set(zip(df["combo"], df["detector"]))
    print(f"Resuming: {len(done)} (combo, detector) pairs already done")
    return df, done


# -----------------------------------------------------------------------------
# Main ablation loop
# -----------------------------------------------------------------------------
def run_ablation():
    train_b, test_b, y_test = load_or_extract()
    print(f"Test anomaly rate: {y_test.mean():.3f}")
    print(f"Train n={len(next(iter(train_b.values())))}, "
          f"Test n={len(y_test)}")
    print(f"Block sizes: " + ", ".join(f"{b}={train_b[b].shape[1]}d"
                                        for b in ATTENTION_BLOCKS))

    attn_combos = [c for r in range(1, len(ATTENTION_BLOCKS) + 1)
                     for c in combinations(ATTENTION_BLOCKS, r)]
    
    # 2. Build the exact list of tests we want to run
    all_combos = [("cls_token_raw",), ("cls_token_pca",)] # The two baselines alone
    all_combos.extend(attn_combos)                        # Attention blocks alone
    for c in attn_combos:
        all_combos.append(("cls_token_pca",) + c)         # PCA token + Attention blocks

    total_fits = sum(len(seeds_for(d)) for d in DETECTOR_NAMES) * len(all_combos)
    print(f"\n{len(all_combos)} specific combos × {len(DETECTOR_NAMES)} detectors = "
          f"{total_fits} total fits")

    existing_df, done = load_existing_results()
    rows = existing_df.to_dict("records") if not existing_df.empty else []

    for i, combo in enumerate(all_combos, 1):
        combo_label = "+".join(combo)

        # Skip combo entirely if all its detectors are done
        if all((combo_label, d) in done for d in DETECTOR_NAMES):
            print(f"[{i:2d}/{len(all_combos)}] {combo_label}  (all done, skipping)")
            continue

        # Build the feature matrix once per combo
        Xtr_parts, Xte_parts = [], []

        # 1. The Paper Baseline (Raw, No Scaler)
        if "cls_token_raw" in combo:
            Xtr_parts.append(train_b["cls_token_raw"])
            Xte_parts.append(test_b["cls_token_raw"])
            
        # 2. The Ablation Baseline (PCA, Standardized)
        if "cls_token_pca" in combo:
            sc_cls = StandardScaler().fit(train_b["cls_token_pca"])
            Xtr_parts.append(sc_cls.transform(train_b["cls_token_pca"]).astype(np.float32))
            Xte_parts.append(sc_cls.transform(test_b["cls_token_pca"]).astype(np.float32))

        # 3. The Attention Features (Standardized)
        attn_blocks = [b for b in combo if b not in ("cls_token_raw", "cls_token_pca")]
        if attn_blocks:
            Xtr_attn = np.concatenate([train_b[b] for b in attn_blocks], axis=1)
            Xte_attn = np.concatenate([test_b[b]  for b in attn_blocks], axis=1)
            
            sc_attn = StandardScaler().fit(Xtr_attn)
            Xtr_parts.append(sc_attn.transform(Xtr_attn).astype(np.float32))
            Xte_parts.append(sc_attn.transform(Xte_attn).astype(np.float32))

        # Combine them back together
        Xtr = np.concatenate(Xtr_parts, axis=1).astype(np.float32)
        Xte = np.concatenate(Xte_parts, axis=1).astype(np.float32)
        print(f"\n[{i:2d}/{len(all_combos)}] {combo_label}  dim={Xtr.shape[1]}")
        
        
        for det_name in DETECTOR_NAMES:
            if (combo_label, det_name) in done:
                continue
            seeds = seeds_for(det_name)

            # --- THIS IS THE CRITICAL TRAINING BLOCK YOU WERE MISSING ---
            aurocs, auprcs, runtimes = [], [], []
            for seed in seeds:
                try:
                    det = make_detector(det_name, seed, Xtr.shape[1])
                    t0 = time.time()
                    det.fit(Xtr)
                    s = det.decision_function(Xte)
                    runtimes.append(time.time() - t0)
                    aurocs.append(roc_auc_score(y_test, s))
                    auprcs.append(average_precision_score(y_test, s))
                except Exception as e:
                    print(f"    {det_name} (seed={seed}) FAILED: "
                          f"{type(e).__name__}: {e}")
                    continue
            # ------------------------------------------------------------

            if not aurocs:
                continue

            row = {
                "combo":      combo_label,
                "n_blocks":   len(combo),
                "dim":        Xtr.shape[1],
                "detector":   det_name,
                "n_seeds":    len(aurocs),
                "auroc_mean": float(np.mean(aurocs)),
                "auroc_std":  float(np.std(aurocs)),
                "auprc_mean": float(np.mean(auprcs)),
                "auprc_std":  float(np.std(auprcs)),
                "time_s":     float(np.mean(runtimes)),
            }
            
            # 1. Add the standard attention blocks
            for b in ATTENTION_BLOCKS:
                row[f"has_{b}"] = int(b in combo)
                
            # 2. Add the two manual baselines BEFORE appending
            row["has_cls_token_raw"] = int("cls_token_raw" in combo)
            row["has_cls_token_pca"] = int("cls_token_pca" in combo)
            
            # 3. Now append the fully built dictionary to the list
            rows.append(row)

            print(f"    {det_name:<10} AUROC {row['auroc_mean']:.4f} ± {row['auroc_std']:.4f}  "
                  f"AUPRC {row['auprc_mean']:.4f} ± {row['auprc_std']:.4f}  "
                  f"({row['time_s']:.1f}s/fit, {row['n_seeds']} seeds)")

            # Save partial after EVERY (combo, detector) so crashes don't lose work
            pd.DataFrame(rows).to_csv(RESULTS_CSV, index=False)

    return pd.DataFrame(rows)


# -----------------------------------------------------------------------------
# Marginal importance per detector
# -----------------------------------------------------------------------------
def feature_importance(df):
    out = []
    for det_name in df["detector"].unique():
        sub = df[df["detector"] == det_name]
        ALL_BLOCKS = ATTENTION_BLOCKS + ["cls_token_raw", "cls_token_pca"]
        for b in ALL_BLOCKS:
            with_b    = sub[sub[f"has_{b}"] == 1]["auroc_mean"]
            without_b = sub[sub[f"has_{b}"] == 0]["auroc_mean"]
            out.append({
                "detector":      det_name,
                "feature":       b,
                "auroc_with":    with_b.mean(),
                "auroc_without": without_b.mean(),
                "delta":         with_b.mean() - without_b.mean(),
                "n_with":        len(with_b),
                "n_without":     len(without_b),
            })
    return pd.DataFrame(out)


# -----------------------------------------------------------------------------
# Run
# -----------------------------------------------------------------------------
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

    # Mean importance across detectors — robust ranking
    print("\n=== Robust ranking (mean delta across detectors) ===")
    pivot["mean_delta"] = pivot.mean(axis=1)
    pivot["agree_positive"] = (pivot[DETECTOR_NAMES] > 0).sum(axis=1)
    print(pivot[["mean_delta", "agree_positive"]].sort_values(
        "mean_delta", ascending=False).round(4).to_string())