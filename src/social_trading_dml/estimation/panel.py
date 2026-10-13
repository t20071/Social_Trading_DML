
"""
Panel transforms for fixed-effects estimation.

- demean_within_trader(): one-way (trader) or two-way (trader + week)
  demeaning by alternating projections
- first_difference(): within-trader first differences (consecutive weeks)
- build_fe_arrays(): the single place that demeans and assembles
  Y, T, W, groups and periods for one specification
"""

import logging
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


def demean_within_trader(
    df: pd.DataFrame,
    cols: List[str],
    group_col: str = "trader_id",
    time_col: Optional[str] = None,
    suffix: str = "_demeaned",
    tol: float = 1e-8,
    max_iter: int = 100,
) -> pd.DataFrame:
    """
    Fixed-effects demeaning.

    With time_col=None: one-way within-trader demeaning (value - trader mean).
    With time_col set: two-way (trader + week) demeaning by alternating
    projections, which is exact in one pass for a balanced panel and converges
    to the two-way within transform for an unbalanced one.

    Creates `{col}{suffix}` for each column in `cols`.

    Parameters
    ----------
    df : pd.DataFrame
        Panel data.
    cols : list of str
        Columns to demean (typically treatment + outcome + confounders).
    group_col : str
        Panel ID column.
    time_col : str, optional
        Time column for two-way demeaning (e.g. 'period').
    suffix : str
        Suffix for the demeaned columns.

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

    values = df[valid_cols].astype(np.float64)
    unit_codes = df[group_col].to_numpy()

    scale = values.std().replace(0, 1.0).to_numpy()
    values = values - values.groupby(unit_codes).transform("mean")
    if time_col is not None:
        time_codes = df[time_col].to_numpy()
        converged = False
        for it in range(max_iter):
            values = values - values.groupby(time_codes).transform("mean")
            unit_means = values.groupby(unit_codes).transform("mean")
            values = values - unit_means
            # relative tolerance: largest remaining unit mean / column SD
            if float((np.abs(unit_means.to_numpy()) / scale).max()) < tol:
                converged = True
                break
        if converged:
            logger.info(f"  Two-way demeaning converged after {it + 1} iteration(s)")
        else:
            logger.warning(
                f"  Two-way demeaning did NOT converge in {max_iter} iterations "
                f"(tol={tol:g}); results are approximate"
            )

    for col in valid_cols:
        df[f"{col}{suffix}"] = values[col].to_numpy()

    fe_label = f"{group_col} + {time_col}" if time_col else group_col
    logger.info(f"  Demeaned {len(valid_cols)} columns by {fe_label} (suffix '{suffix}')")

    return df


def first_difference(
    df: pd.DataFrame,
    cols: List[str],
    group_col: str = "trader_id",
    time_col: str = "period",
    suffix: str = "_fd",
) -> pd.DataFrame:
    """
    Within-trader first differences, `{col}{suffix}` = x_t - x_{t-1}.

    Only differences between consecutive periods are kept; the first row of
    each trader and rows after a gap get NaN.
    """
    df = df.sort_values([group_col, time_col]).copy()
    g = df.groupby(group_col)
    consecutive = (df[time_col] - g[time_col].shift(1)) == 1
    for col in cols:
        df[f"{col}{suffix}"] = (df[col] - g[col].shift(1)).where(consecutive)
    return df


def build_fe_arrays(
    df: pd.DataFrame,
    treatment_col: str,
    outcome_col: str,
    control_cols: List[str],
    fe: str = "twoway",
    group_col: str = "trader_id",
    time_col: str = "period",
) -> Dict[str, Any]:
    """
    Demean treatment, outcome and controls and return model arrays.

    fe : 'twoway' (trader + week), 'trader', 'week' (one-way) or 'none' (levels).
    Controls absorbed by the fixed effects (SD after demeaning below 1e-6 of
    the SD before, e.g. tenure under two-way FE) are dropped.
    """
    control_cols = list(dict.fromkeys(control_cols))
    cols = list(dict.fromkeys([treatment_col, outcome_col] + control_cols))
    sub = df[[group_col, time_col] + cols].dropna(subset=cols)

    if fe == "none":
        vals = sub[cols].astype(np.float64)
    elif fe == "week":
        dm = demean_within_trader(sub, cols, group_col=time_col, suffix="__dm")
        vals = dm[[f"{c}__dm" for c in cols]]
        vals.columns = cols
    else:
        dm = demean_within_trader(
            sub, cols, group_col=group_col,
            time_col=time_col if fe == "twoway" else None, suffix="__dm",
        )
        vals = dm[[f"{c}__dm" for c in cols]]
        vals.columns = cols

    W_df = vals[control_cols]
    raw_sd = sub[control_cols].std()
    keep = [c for c in control_cols
            if raw_sd[c] > 0 and W_df[c].std() > 1e-6 * raw_sd[c]]
    dropped = sorted(set(control_cols) - set(keep))
    if dropped:
        logger.info(f"  Dropping controls with no variation after FE: {dropped}")

    return {
        "Y": vals[outcome_col].to_numpy(),
        "T": vals[treatment_col].to_numpy(),
        "W": vals[keep].to_numpy(),
        "W_names": keep,
        "groups": sub[group_col].to_numpy(),
        "periods": sub[time_col].to_numpy(),
        "index": sub.index,
    }
