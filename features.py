"""BERT attention extraction and feature caching."""

import os
import gc

import numpy as np
import torch
import transformers
transformers.logging.set_verbosity_error()
from transformers import AutoTokenizer, AutoModel
from datasets import load_dataset
from sklearn.decomposition import PCA
from huggingface_hub import hf_hub_download

from config import HF_REPO, MODEL_NAME, BATCH_SIZE, DEVICE, ATTENTION_BLOCKS, RunConfig


def load_split(filename):
    ds = load_dataset(HF_REPO, data_files=filename, split="train")
    return list(ds["text"]), np.array(ds["label"])


def load_paper_cls_token(subdataset: str):
    """Download the paper's precomputed bert-base-uncased CLS embeddings."""
    train_path = hf_hub_download(
        repo_id=HF_REPO, repo_type="dataset",
        filename=f"embeddings/{subdataset}/{subdataset}_train_data_bert_base_uncased_feature.npy",
    )
    test_path = hf_hub_download(
        repo_id=HF_REPO, repo_type="dataset",
        filename=f"embeddings/{subdataset}/{subdataset}_test_data_bert_base_uncased_feature.npy",
    )
    return np.load(train_path), np.load(test_path)


@torch.no_grad()
def extract_attention_blocks(texts, tokenizer, model, layer_idx, max_len: int):
    """Extract five attention statistics from the given BERT layer."""
    blocks = {b: [] for b in ATTENTION_BLOCKS}
    for start in range(0, len(texts), BATCH_SIZE):
        batch = texts[start:start + BATCH_SIZE]
        enc  = tokenizer(batch, padding="max_length", truncation=True,
                         max_length=max_len, return_tensors="pt").to(DEVICE)
        out  = model(**enc)
        attn = out.attentions[layer_idx]
        mask  = enc["attention_mask"].float()
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

    return {b: np.concatenate(v, axis=0).astype(np.float32) for b, v in blocks.items()}


def load_or_extract(run_cfg: RunConfig):
    """Return (train_blocks, test_blocks, y_test), loading from cache or extracting fresh."""
    all_features = ATTENTION_BLOCKS + ["cls_token_raw", "cls_token_pca"]
    subdataset   = run_cfg.subdataset

    if os.path.exists(run_cfg.feature_file):
        print(f"Loading cached features from {run_cfg.feature_file}")
        d = np.load(run_cfg.feature_file)
        return (
            {b: d[f"train_{b}"] for b in all_features},
            {b: d[f"test_{b}"]  for b in all_features},
            d["test_labels"],
        )

    print("Extracting attention features fresh...")
    train_texts, _     = load_split(f"datasets/{subdataset}/{subdataset}_train_data.jsonl")
    test_texts, y_test = load_split(f"datasets/{subdataset}/{subdataset}_test_data.jsonl")

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModel.from_pretrained(
        MODEL_NAME, output_attentions=True, attn_implementation="eager",
    ).to(DEVICE).eval()
    layer_idx = model.config.num_hidden_layers - 1

    print("Train features:")
    train_b = extract_attention_blocks(train_texts, tokenizer, model, layer_idx, run_cfg.max_len)
    print("Test features:")
    test_b  = extract_attention_blocks(test_texts,  tokenizer, model, layer_idx, run_cfg.max_len)

    print("Loading paper's precomputed cls_token embeddings...")
    paper_train_cls, paper_test_cls = load_paper_cls_token(subdataset)

    n_train = len(next(iter(train_b.values())))
    n_test  = len(next(iter(test_b.values())))
    assert paper_train_cls.shape[0] == n_train
    assert paper_test_cls.shape[0]  == n_test

    train_b["cls_token_raw"] = paper_train_cls.astype(np.float32)
    test_b["cls_token_raw"]  = paper_test_cls.astype(np.float32)

    pca = PCA(n_components=64, random_state=42)
    train_b["cls_token_pca"] = pca.fit_transform(paper_train_cls).astype(np.float32)
    test_b["cls_token_pca"]  = pca.transform(paper_test_cls).astype(np.float32)
    print(f"PCA(64) explained variance: {pca.explained_variance_ratio_.sum():.3f}")

    save = {f"train_{b}": train_b[b] for b in all_features}
    save.update({f"test_{b}": test_b[b] for b in all_features})
    save["test_labels"] = y_test
    np.savez_compressed(run_cfg.feature_file, **save)
    print(f"Saved features to {run_cfg.feature_file}")

    del model, tokenizer
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    gc.collect()

    return train_b, test_b, y_test
