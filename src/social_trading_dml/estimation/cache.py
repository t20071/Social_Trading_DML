
"""
Cache for fitted models and cross-fitted residuals.

Files are named ``{cache_name}_{fingerprint}.pkl`` in config.CACHE_DIR. The
fingerprint hashes every input array (bytes, shape, dtype), the fit settings,
the nuisance-learner definitions, CACHE_VERSION and the versions of the
libraries that produce the fitted models. Summary statistics are recomputed
from the cached residuals on every load, so they never go stale.
"""

import hashlib
import logging
import os
import pickle
from typing import Optional

import numpy as np

from social_trading_dml.config import CACHE_DIR as _CACHE_DIR

logger = logging.getLogger(__name__)

CACHE_DIR = str(_CACHE_DIR)

# Bump when nuisance fitting or fold construction changes. Final-stage
# summaries (estimate, SEs) are recomputed from cached residuals on load, so
# changes to clustered_final_stage do not need a bump.
CACHE_VERSION = 3


def env_signature() -> str:
    """Versions of the libraries that determine fitted nuisance models."""
    import sklearn
    try:
        import econml
        econml_v = econml.__version__
    except ImportError:
        econml_v = "none"
    try:
        import lightgbm
        lgb_v = lightgbm.__version__
    except ImportError:
        lgb_v = "none"
    return f"np{np.__version__}|skl{sklearn.__version__}|econml{econml_v}|lgb{lgb_v}"


def fingerprint(*arrays, **params) -> str:
    """
    Short hash of the input arrays, fit settings, cache version and library
    versions, used in cache file names.
    """
    h = hashlib.sha1(f"v{CACHE_VERSION}|{env_signature()}".encode())
    for a in arrays:
        if a is None:
            h.update(b"none")
            continue
        a = np.ascontiguousarray(a)
        h.update(str((a.shape, a.dtype.str)).encode())
        h.update(a.tobytes())
    h.update(repr(sorted(params.items())).encode())
    return h.hexdigest()[:12]


def set_cache_dir(path) -> None:
    """Use another cache directory (e.g. for smoke runs)."""
    global CACHE_DIR
    CACHE_DIR = str(path)


def _find(fp: str) -> Optional[str]:
    """Any cache file with this fingerprint, whatever name it was saved under."""
    if not os.path.isdir(CACHE_DIR):
        return None
    for name in os.listdir(CACHE_DIR):
        if name.endswith(f"_{fp}.pkl"):
            return os.path.join(CACHE_DIR, name)
    return None


def load_cache(cache_name: str, fingerprint: str) -> Optional[dict]:
    """
    Load a cached fit. The fingerprint identifies the fit, so a file saved
    under another cache name with the same fingerprint is reused.
    """
    cache_path = _find(fingerprint)
    if cache_path:
        logger.info(f"Loading cached model from {cache_path}")
        with open(cache_path, "rb") as f:
            return pickle.load(f)
    return None


def save_cache(cache_name: str, obj: dict) -> None:
    """
    Save a fitted model/result as `{cache_name}.pkl` (cache_name ends with the
    fingerprint). Skipped if a file with the same fingerprint already exists.
    """
    fp = cache_name.rsplit("_", 1)[-1]
    if _find(fp):
        return
    cache_path = os.path.join(CACHE_DIR, f"{cache_name}.pkl")
    os.makedirs(CACHE_DIR, exist_ok=True)
    with open(cache_path, "wb") as f:
        pickle.dump(obj, f)
    logger.info(f"Cached to {cache_path}")
