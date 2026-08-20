"""
Treatment and outcome variable construction.

Treatment (T): total_self_trxn (raw count — NOT the ratio as primary)
    - Primary spec: raw count of self-trades that week
    - Robustness: self_trade_ratio = total_self_trxn / total_trxn
    - The full panel is used INCLUDING inactive trader-weeks (total_trxn==0)

Outcome (Y): total_gain
    - Primary: winsorized at 1st/99th percentiles
    - Robustness: asinh(total_gain)
    - NEVER plain log() — total_gain takes negative values

Also implements:
    - Within-trader demeaning for fixed-effects DML
    - Trader-level cross-section collapse for triangulation
"""

import logging
from typing import Dict, Any, Tuple, Optional, List

import pandas as pd
import numpy as np

logger = logging.getLogger(__name__)


def compute_treatment(df: pd.DataFrame) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """
    Construct treatment variables.

    Primary treatment (T): total_self_trxn (raw count)
    Robustness variants:
        - self_trade_ratio = total_self_trxn / total_trxn  (NaN when total_trxn==0)
        - self_trade_count_log = log1p(total_self_trxn)

    NOTE: Per the spec, we do NOT filter rows where total_trxn <= 0.
    The model runs on the FULL panel including inactive trader-weeks.

    Parameters
    ----------
    df : pd.DataFrame
        Must contain 'total_self_trxn' and 'total_trxn'.

    Returns
    -------
    Tuple[pd.DataFrame, Dict]
        DataFrame with treatment columns added, and descriptive stats.
    """
    df = df.copy()
    stats = {}

    stats["total_rows"] = len(df)

    # Active vs inactive weeks
    active_mask = df["total_trxn"] > 0
    stats["active_weeks"] = int(active_mask.sum())
    stats["inactive_weeks"] = int((~active_mask).sum())
    stats["active_fraction"] = float(active_mask.mean())

    logger.info(
        f"Active weeks (total_trxn > 0): {stats['active_weeks']:,} "
        f"({stats['active_fraction']:.1%}) — "
        f"Inactive: {stats['inactive_weeks']:,}"
    )

    # Robustness: ratio (NaN for inactive weeks)
    df["self_trade_ratio"] = np.where(
        active_mask,
        df["total_self_trxn"] / df["total_trxn"],
        np.nan,
    )

    # Robustness: log-count
    df["self_trade_count_log"] = np.log1p(df["total_self_trxn"])

    # Treatment distribution on full panel
    t = df["total_self_trxn"]
    stats["treatment_mean"] = float(t.mean())
    stats["treatment_median"] = float(t.median())
    stats["treatment_std"] = float(t.std())
    stats["treatment_zero_share"] = float((t == 0).mean())

    # Also report ratio stats on active weeks
    active_ratio = df.loc[active_mask, "self_trade_ratio"]
    stats["ratio_mean_active"] = float(active_ratio.mean())
    stats["ratio_median_active"] = float(active_ratio.median())
    stats["ratio_zero_share_active"] = float((active_ratio == 0).mean())
    stats["ratio_one_share_active"] = float((active_ratio == 1).mean())

    logger.info(
        f"Treatment (total_self_trxn) full panel: "
        f"mean={stats['treatment_mean']:.3f}, "
        f"median={stats['treatment_median']:.1f}, "
        f"zero share={stats['treatment_zero_share']:.1%}"
    )

    return df, stats


def compute_outcome(
    df: pd.DataFrame,
    winsorize_pcts: Tuple[float, float] = (0.01, 0.99),
) -> pd.DataFrame:
    """
    Construct outcome variables:
    - total_gain_w: winsorized total_gain at given percentiles
    - total_gain_asinh: asinh(total_gain) for robustness

    Parameters
    ----------
    df : pd.DataFrame
        Must contain 'total_gain'.
    winsorize_pcts : tuple of (lower, upper)
        Percentiles for winsorization (default: 1st/99th).

    Returns
    -------
    pd.DataFrame
        DataFrame with outcome columns added.
    """
    df = df.copy()

    lower_bound = df["total_gain"].quantile(winsorize_pcts[0])
    upper_bound = df["total_gain"].quantile(winsorize_pcts[1])
    df["total_gain_w"] = df["total_gain"].clip(lower=lower_bound, upper=upper_bound)

    logger.info(
        f"Winsorized total_gain at [{winsorize_pcts[0]:.0%}, "
        f"{winsorize_pcts[1]:.0%}]: bounds = [{lower_bound:.2f}, {upper_bound:.2f}]"
    )

    # Asinh outcome (handles negatives, unlike log)
    df["total_gain_asinh"] = np.arcsinh(df["total_gain"])

    return df


def demean_within_trader(
    df: pd.DataFrame,
    cols: List[str],
    group_col: str = "trader_id",
) -> pd.DataFrame:
    """
    Within-trader demeaning for fixed-effects specification.

    For each column in `cols`, creates `{col}_demeaned` = value - trader_mean.
    This absorbs trader-level unobserved heterogeneity (skill, risk appetite).

    Parameters
    ----------
    df : pd.DataFrame
        Panel data.
    cols : list of str
        Columns to demean (typically treatment + outcome + confounders).
    group_col : str
        Panel ID column.

    Returns
    -------
    pd.DataFrame
        DataFrame with demeaned columns added.
    """
    df = df.copy()

    # Only demean columns that exist and have numeric dtype
    valid_cols = [c for c in cols if c in df.columns and df[c].dtype.kind in "iufb"]
    if len(valid_cols) < len(cols):
        skipped = set(cols) - set(valid_cols)
        logger.warning(f"Skipping demeaning for non-numeric/missing columns: {skipped}")

    trader_means = df.groupby(group_col)[valid_cols].transform("mean")

    for col in valid_cols:
        demeaned_name = f"{col}_demeaned"
        df[demeaned_name] = df[col] - trader_means[col]
        mean_check = abs(df[demeaned_name].mean())
        logger.info(
            f"  Demeaned '{col}' → '{demeaned_name}' "
            f"(mean of demeaned: {mean_check:.6f})"
        )

    return df


def collapse_to_trader_level(
    df: pd.DataFrame,
    confounder_cols: List[str],
    treatment_col: str = "total_self_trxn",
    outcome_col: str = "total_gain_w",
    group_col: str = "trader_id",
) -> pd.DataFrame:
    """
    Collapse panel to one row per trader for cross-sectional triangulation.

    Uses each trader's EARLIEST available confounder values as controls
    (not means — earliest is less contaminated by treatment feedback).
    Treatment and outcome are averaged over the trader's periods.

    Parameters
    ----------
    df : pd.DataFrame
        Panel data.
    confounder_cols : list of str
        Confounder columns (lagged versions).
    treatment_col : str
        Treatment column.
    outcome_col : str
        Outcome column.
    group_col : str
        Panel ID column.

    Returns
    -------
    pd.DataFrame
        One row per trader.
    """
    df = df.sort_values([group_col, "period"]).copy()

    # Earliest observation confounders
    # Only pick columns that exist in df
    existing_conf = [c for c in confounder_cols if c in df.columns]
    earliest = df.groupby(group_col).first()[existing_conf].reset_index()

    # Mean treatment and outcome over all periods
    agg = df.groupby(group_col).agg(
        **{
            treatment_col: (treatment_col, "mean"),
            outcome_col: (outcome_col, "mean"),
            "n_periods": ("period", "count"),
        }
    ).reset_index()

    collapsed = earliest.merge(agg, on=group_col)

    logger.info(
        f"Collapsed panel to trader level: {len(collapsed):,} traders, "
        f"{collapsed.shape[1]} columns"
    )

    return collapsed


def get_analysis_sample(
    df: pd.DataFrame,
    treatment_col: str = "total_self_trxn",
    confounder_lag_cols: Optional[List[str]] = None,
    require_tenure: bool = True,
) -> Tuple[pd.DataFrame, Dict[str, int]]:
    """
    Build the analysis sample with explicit, narrow dropna steps.

    Steps (each documented separately for the report):
    1. Require non-null treatment (total_self_trxn — should be 0 nulls)
    2. Require non-null outcome (total_gain — should be 0 nulls)
    3. Require non-null lagged confounders (drops first period per trader)
    4. Optionally require non-null tenure (drops ~20K rows with missing joining_date)

    CRITICAL: No blanket dropna(). Each step targets an explicit, narrow
    subset of columns and logs its row-count impact separately.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame with treatment, outcome, and lagged confounders.
    treatment_col : str
        Treatment column name.
    confounder_lag_cols : list of str, optional
        Lagged confounder columns to require non-missing.
    require_tenure : bool
        Whether to require non-null tenure_weeks.

    Returns
    -------
    Tuple[pd.DataFrame, Dict]
        Filtered DataFrame and sample construction counts.
    """
    counts = {}
    counts["total"] = len(df)

    # Step 1: Non-null treatment
    mask = df[treatment_col].notna()
    counts["after_treatment_filter"] = int(mask.sum())
    counts["dropped_null_treatment"] = counts["total"] - counts["after_treatment_filter"]
    logger.info(
        f"  Step 1 (treatment not null): {counts['total']:,} → "
        f"{counts['after_treatment_filter']:,} "
        f"(dropped {counts['dropped_null_treatment']:,})"
    )

    # Step 2: Non-null outcome
    mask = mask & df["total_gain"].notna()
    counts["after_outcome_filter"] = int(mask.sum())
    counts["dropped_null_outcome"] = (
        counts["after_treatment_filter"] - counts["after_outcome_filter"]
    )

    # Step 3: Non-null lagged confounders
    if confounder_lag_cols:
        for col in confounder_lag_cols:
            if col in df.columns:
                before = int(mask.sum())
                mask = mask & df[col].notna()
                after = int(mask.sum())
                if before - after > 0:
                    logger.info(
                        f"  Step 3 (lag {col} not null): dropped {before - after:,}"
                    )
    counts["after_lag_filter"] = int(mask.sum())
    counts["dropped_missing_lags"] = (
        counts["after_outcome_filter"] - counts["after_lag_filter"]
    )

    # Step 4: Non-null tenure (optional)
    if require_tenure and "tenure_weeks" in df.columns:
        before = int(mask.sum())
        mask = mask & df["tenure_weeks"].notna()
        counts["after_tenure_filter"] = int(mask.sum())
        counts["dropped_missing_tenure"] = before - counts["after_tenure_filter"]
        logger.info(
            f"  Step 4 (tenure not null): dropped "
            f"{counts['dropped_missing_tenure']:,}"
        )
    else:
        counts["after_tenure_filter"] = int(mask.sum())
        counts["dropped_missing_tenure"] = 0

    df_filtered = df[mask].copy()
    counts["final_rows"] = len(df_filtered)
    counts["final_traders"] = int(df_filtered["trader_id"].nunique())

    logger.info(
        f"Sample construction: {counts['total']:,} → "
        f"{counts['final_rows']:,} "
        f"({counts['final_traders']:,} traders)"
    )

    return df_filtered, counts
