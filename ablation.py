"""Ablation loop, feature matrix assembly, and marginal importance analysis."""

import os
import time
from itertools import combinations

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score, average_precision_score

from config import ATTENTION_BLOCKS, DETECTOR_NAMES, RunConfig
from features import load_or_extract
from detectors import make_detector, seeds_for


def load_existing_results(results_csv: str):
    if not os.path.exists(results_csv):
        return pd.DataFrame(), set()
    df = pd.read_csv(results_csv)
    done = set(zip(df["combo"], df["detector"]))
    print(f"Resuming: {len(done)} (combo, detector) pairs already done")
    return df, done


def build_feature_matrix(combo, train_b, test_b):
    """Assemble and scale the feature matrix for a given feature combination.

    - cls_token_raw is passed through raw (StandardScaler distorts BERT geometry).
    - cls_token_pca and attention blocks are standardized independently.
    """
    Xtr_parts, Xte_parts = [], []

    if "cls_token_raw" in combo:
        Xtr_parts.append(train_b["cls_token_raw"])
        Xte_parts.append(test_b["cls_token_raw"])

    if "cls_token_pca" in combo:
        sc = StandardScaler().fit(train_b["cls_token_pca"])
        Xtr_parts.append(sc.transform(train_b["cls_token_pca"]).astype(np.float32))
        Xte_parts.append(sc.transform(test_b["cls_token_pca"]).astype(np.float32))

    attn_blocks = [b for b in combo if b not in ("cls_token_raw", "cls_token_pca")]
    if attn_blocks:
        Xtr_attn = np.concatenate([train_b[b] for b in attn_blocks], axis=1)
        Xte_attn = np.concatenate([test_b[b]  for b in attn_blocks], axis=1)
        sc = StandardScaler().fit(Xtr_attn)
        Xtr_parts.append(sc.transform(Xtr_attn).astype(np.float32))
        Xte_parts.append(sc.transform(Xte_attn).astype(np.float32))

    return (
        np.concatenate(Xtr_parts, axis=1).astype(np.float32),
        np.concatenate(Xte_parts, axis=1).astype(np.float32),
    )


def _eval_detector(det_name, Xtr, Xte, y_test):
    """Fit a detector across all its seeds; return (aurocs, auprcs, runtimes)."""
    aurocs, auprcs, runtimes = [], [], []
    for seed in seeds_for(det_name):
        try:
            det = make_detector(det_name, seed, Xtr.shape[1])
            t0  = time.time()
            det.fit(Xtr)
            s   = det.decision_function(Xte)
            runtimes.append(time.time() - t0)
            aurocs.append(roc_auc_score(y_test, s))
            auprcs.append(average_precision_score(y_test, s))
        except Exception as e:
            print(f"    {det_name} (seed={seed}) FAILED: {type(e).__name__}: {e}")
    return aurocs, auprcs, runtimes


def run_ablation(run_cfg: RunConfig) -> pd.DataFrame:
    """Run the full ablation and return a DataFrame of results."""
    train_b, test_b, y_test = load_or_extract(run_cfg)
    print(f"Test anomaly rate: {y_test.mean():.3f}")
    print(f"Train n={len(next(iter(train_b.values())))}, Test n={len(y_test)}")
    print("Block sizes: " + ", ".join(f"{b}={train_b[b].shape[1]}d" for b in ATTENTION_BLOCKS))

    attn_combos = [c for r in range(1, len(ATTENTION_BLOCKS) + 1)
                     for c in combinations(ATTENTION_BLOCKS, r)]
    all_combos = [("cls_token_raw",), ("cls_token_pca",)]
    all_combos.extend(attn_combos)
    for c in attn_combos:
        all_combos.append(("cls_token_pca",) + c)

    total_fits = sum(len(seeds_for(d)) for d in DETECTOR_NAMES) * len(all_combos)
    print(f"\n{len(all_combos)} combos × {len(DETECTOR_NAMES)} detectors = {total_fits} total fits")

    existing_df, done = load_existing_results(run_cfg.results_csv)
    rows = existing_df.to_dict("records") if not existing_df.empty else []

    for i, combo in enumerate(all_combos, 1):
        combo_label = "+".join(combo)

        if all((combo_label, d) in done for d in DETECTOR_NAMES):
            print(f"[{i:2d}/{len(all_combos)}] {combo_label}  (all done, skipping)")
            continue

        Xtr, Xte = build_feature_matrix(combo, train_b, test_b)
        print(f"\n[{i:2d}/{len(all_combos)}] {combo_label}  dim={Xtr.shape[1]}")

        for det_name in DETECTOR_NAMES:
            if (combo_label, det_name) in done:
                continue

            aurocs, auprcs, runtimes = _eval_detector(det_name, Xtr, Xte, y_test)
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
            for b in ATTENTION_BLOCKS:
                row[f"has_{b}"] = int(b in combo)
            row["has_cls_token_raw"] = int("cls_token_raw" in combo)
            row["has_cls_token_pca"] = int("cls_token_pca" in combo)
            rows.append(row)

            print(f"    {det_name:<10} AUROC {row['auroc_mean']:.4f} ± {row['auroc_std']:.4f}  "
                  f"AUPRC {row['auprc_mean']:.4f} ± {row['auprc_std']:.4f}  "
                  f"({row['time_s']:.1f}s/fit, {row['n_seeds']} seeds)")

            pd.DataFrame(rows).to_csv(run_cfg.results_csv, index=False)

    return pd.DataFrame(rows)


def feature_importance(df):
    """Compute marginal AUROC contribution of each feature per detector."""
    ALL_BLOCKS = ATTENTION_BLOCKS + ["cls_token_raw", "cls_token_pca"]
    out = []
    for det_name in df["detector"].unique():
        sub = df[df["detector"] == det_name]
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
