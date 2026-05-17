"""Anomaly detector factory."""

from pyod.models.lof          import LOF
from pyod.models.deep_svdd    import DeepSVDD
from pyod.models.ecod         import ECOD
from pyod.models.iforest      import IForest
from pyod.models.so_gaal      import SO_GAAL
from pyod.models.auto_encoder import AutoEncoder
from pyod.models.vae          import VAE
from pyod.models.lunar        import LUNAR

from config import DETERMINISTIC, STOCHASTIC_SEEDS


def make_detector(name, seed, input_dim):
    if name == "LOF":      return LOF(n_jobs=-1)
    if name == "ECOD":     return ECOD(n_jobs=-1)
    if name == "LUNAR":    return LUNAR()
    if name == "SO-GAAL":  return SO_GAAL()
    if name == "IForest":  return IForest(random_state=seed, n_jobs=-1)
    if name == "DeepSVDD": return DeepSVDD(n_features=input_dim, random_state=seed, verbose=0)
    if name == "AE":       return AutoEncoder(random_state=seed, verbose=0)
    if name == "VAE":      return VAE(random_state=seed, verbose=0)
    raise ValueError(f"Unknown detector: {name}")


def seeds_for(name):
    return (0,) if name in DETERMINISTIC else STOCHASTIC_SEEDS
