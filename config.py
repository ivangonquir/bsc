from dataclasses import dataclass
import torch

HF_REPO    = "kendx/NLP-ADBench"
MODEL_NAME = "bert-base-uncased"
BATCH_SIZE = 16
DEVICE     = "cuda" if torch.cuda.is_available() else "cpu"

ATTENTION_BLOCKS = ["cls_mean", "cls_max", "cls_std", "head_entropy", "diag_mean"]

# Per-dataset settings (max sequence length matched to the original paper).
DATASET_CONFIG = {
    "sms_spam":   {"max_len": 32},
    "email_spam": {"max_len": 256},
    "bbc":        {"max_len": 512},
}

# Neural-network detectors that are impractically slow without a GPU.
GPU_HEAVY_DETECTORS = {"DeepSVDD", "AE", "VAE", "LUNAR", "SO-GAAL"}
_ALL_DETECTOR_NAMES = ["LOF", "DeepSVDD", "ECOD", "IForest", "SO-GAAL", "AE", "VAE", "LUNAR"]

if DEVICE == "cpu":
    DETECTOR_NAMES = [d for d in _ALL_DETECTOR_NAMES if d not in GPU_HEAVY_DETECTORS]
    print(
        f"[WARNING] No CUDA GPU detected — skipping GPU-heavy detectors: "
        f"{sorted(GPU_HEAVY_DETECTORS)}.\n"
        f"          Running CPU-only detectors: {DETECTOR_NAMES}"
    )
else:
    DETECTOR_NAMES = _ALL_DETECTOR_NAMES

DETERMINISTIC    = {"LOF", "ECOD", "LUNAR", "SO-GAAL"}
STOCHASTIC_SEEDS = (0, 1, 2)

# Mapping from dataset name to its results directory.
DATASET_DIRS = {
    "sms_spam":   "sms-spam",
    "email_spam": "email_spam-out",
    "bbc":        "bbc-out",
}


@dataclass
class RunConfig:
    """All settings that vary per dataset run."""
    subdataset:   str
    max_len:      int
    feature_file: str
    results_csv:  str
    importance_csv: str

    @classmethod
    def from_dataset(cls, dataset: str) -> "RunConfig":
        if dataset not in DATASET_CONFIG:
            raise ValueError(f"Unknown dataset '{dataset}'. "
                             f"Choose from: {list(DATASET_CONFIG)}")
        max_len = DATASET_CONFIG[dataset]["max_len"]
        n_blocks = len(ATTENTION_BLOCKS)
        return cls(
            subdataset    = dataset,
            max_len       = max_len,
            feature_file  = f"{dataset}_features_{n_blocks}blocks_plus_cls_maxlen{max_len}.npz",
            results_csv   = f"{dataset}_attn_ablation_all_detectors.csv",
            importance_csv= f"{dataset}_attn_importance_all_detectors.csv",
        )
