import torch

HF_REPO    = "kendx/NLP-ADBench"
MODEL_NAME = "bert-base-uncased"
SUBDATASET = "sms_spam"
MAX_LEN    = 32
BATCH_SIZE = 16
DEVICE     = "cuda" if torch.cuda.is_available() else "cpu"

ATTENTION_BLOCKS = ["cls_mean", "cls_max", "cls_std", "head_entropy", "diag_mean"]

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

FEATURE_FILE   = f"{SUBDATASET}_features_{len(ATTENTION_BLOCKS)}blocks_plus_cls_maxlen{MAX_LEN}.npz"
RESULTS_CSV    = f"{SUBDATASET}_attn_ablation_all_detectors.csv"
IMPORTANCE_CSV = f"{SUBDATASET}_attn_importance_all_detectors.csv"

# Mapping from dataset name to its results directory
DATASET_DIRS = {
    "sms_spam":   "sms-spam",
    "email_spam": "email_spam-out",
    "bbc":        "bbc-out",
}
