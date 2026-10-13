
"""
Project configuration: paths, core variable names and analysis constants.

The pipeline, figures, notebooks and tests all read locations and variable
names from here, so there is a single definition of each.
"""

import os
from pathlib import Path
from typing import Dict, List, Tuple

# Repo root. Defaults to the source checkout (editable install); set
# SOCIAL_TRADING_DML_ROOT to use another working directory.
PROJECT_ROOT = Path(os.environ.get("SOCIAL_TRADING_DML_ROOT",
                                   Path(__file__).resolve().parents[2]))

# ── Data (local only, gitignored) ──────────────────────────────────────────
DATA_DIR = PROJECT_ROOT / "data"
RAW_DATA_CANDIDATES = [
    DATA_DIR / "raw" / "trader_data_exl.csv",
    PROJECT_ROOT / "trader_data_exl.csv",  # legacy location
]
PROCESSED_DIR = DATA_DIR / "processed"   # cleaned panel (parquet) + cleaning stats
PANEL_PARQUET = PROCESSED_DIR / "trader_panel.parquet"
CLEANING_STATS_JSON = PROCESSED_DIR / "cleaning_stats.json"
CACHE_DIR = DATA_DIR / "cache"           # fitted models + cross-fitted residuals

# ── Outputs (tracked) ──────────────────────────────────────────────────────
RESULTS_DIR = PROJECT_ROOT / "results"               # machine-readable results
FIGURES_DIR = PROJECT_ROOT / "reports" / "figures"   # figures used by reports/report.md

# ── Variables ──────────────────────────────────────────────────────────────
TREATMENT = "total_self_trxn"
OUTCOME = "total_gain_w"
# Effect modifiers for the causal forest. All trader-level ones are fixed
# before the analysis window (period 30), so they cannot respond to the
# treatment; `period` lets the forest separate calendar from trader
# heterogeneity.
HETEROGENEITY_COLS: List[str] = [
    "join_cohort_weeks",   # weeks on the platform at the start of the window
    "copiers_start",       # copier audience in period 30
    "followers_start",     # follower audience in period 30
    "period",
]
TRADER_MODERATORS: List[str] = ["join_cohort_weeks", "copiers_start", "followers_start"]

# Period blocks (one period = one week; period 30 = 1-7 Jan 2013,
# period 75 = 12-18 Nov 2013). The 53/54 boundary is the best single break
# in the week-specific effects (inverse-variance weighted); the 72/73
# boundary is where activity jumps. Used for block refits, within-block
# GATES and the BLP test.
WEEK_BLOCKS: Dict[str, Tuple[int, int]] = {
    "periods 31-53": (31, 53),
    "periods 54-72": (54, 72),
    "periods 73-75": (73, 75),
}
LATE_BLOCK_START = 73          # first period of the final, most active block

RANDOM_STATE = 42
HALF_SPLIT_SEED = 7            # random half-split of traders for the forest
MIN_CLUSTERS_FOR_CRVE = 10     # fewer week clusters -> two-way CIs not reported
WILD_BOOTSTRAP_REPS = 9999


def default_raw_data_path() -> Path:
    """First existing raw-data location (data/raw/ preferred), else data/raw/."""
    for path in RAW_DATA_CANDIDATES:
        if path.exists():
            return path
    return RAW_DATA_CANDIDATES[0]
