"""
Treatment, outcome and analysis-sample construction.

Treatment (T): total_self_trxn — count of self-initiated trades in the week
    - Robustness: self_trade_ratio = total_self_trxn / total_trxn (active weeks)
      and log1p(total_self_trxn)
    - The full panel is used, including inactive trader-weeks

Outcome (Y): total_gain (weekly realized P&L, in dataset units)
    - Primary: winsorized at the 1st/99th percentiles of its NON-ZERO values
      (gains are exactly 0 in ~94% of weeks; percentiles over all weeks clip a
      large share of real P&L — the v1 bug — and the same holds for the
      self/copied components within active weeks)
    - Robustness: asinh(total_gain), raw total_gain
    - Decomposition: total_gain == total_self_gain + total_copied_gain, so the
      effect on total gain is largely the P&L of the extra self-trade itself
    - Never plain log(): gains take negative values

Fixed-effects transforms (demeaning, first differences) live in
social_trading_dml.estimation.panel.
"""

import logging
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from social_trading_dml.config import OUTCOME, TREATMENT

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
    - total_gain_w: total_gain winsorized at percentiles of its non-zero values
    - total_gain_asinh: asinh(total_gain) for robustness
    - total_self_gain_w / total_copied_gain_w: each component winsorized at
      percentiles of ITS OWN non-zero values (self gain is zero in most active
      weeks, so active-week percentiles would clip it far too tightly)

    Parameters
    ----------
    df : pd.DataFrame
        Must contain 'total_gain'.
    winsorize_pcts : tuple of (lower, upper)
        Percentiles for winsorization (default: 1st/99th).

    Returns
    -------
    pd.DataFrame
        DataFrame with outcome columns added; df.attrs['winsor_bounds'] holds
        {col: (lower, upper, share_of_nonzero_clipped)}.
    """
    df = df.copy()
    bounds = {}

    for col in ["total_gain", "total_self_gain", "total_copied_gain"]:
        if col not in df.columns:
            continue
        nz = df.loc[df[col] != 0, col]
        lower_bound = nz.quantile(winsorize_pcts[0])
        upper_bound = nz.quantile(winsorize_pcts[1])
        df[f"{col}_w"] = df[col].clip(lower=lower_bound, upper=upper_bound)
        share_clipped = float(((nz < lower_bound) | (nz > upper_bound)).mean())
        bounds[col] = (float(lower_bound), float(upper_bound), share_clipped)
        logger.info(
            f"Winsorized {col} at non-zero [{winsorize_pcts[0]:.0%}, "
            f"{winsorize_pcts[1]:.0%}]: bounds = [{lower_bound:.2f}, {upper_bound:.2f}], "
            f"{share_clipped:.1%} of non-zero weeks clipped"
        )
    df.attrs["winsor_bounds"] = bounds

    # Asinh outcome (handles negatives, unlike log)
    df["total_gain_asinh"] = np.arcsinh(df["total_gain"])

    return df


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
    4. Optionally require non-null tenure (rows with missing joining_date)
    5. Drop pre-join weeks (pre_join == True), if the flag is present

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

    # Step 5: Drop weeks before the trader joined the platform
    if "pre_join" in df.columns:
        before = int(mask.sum())
        mask = mask & ~df["pre_join"].fillna(False).astype(bool)
        counts["dropped_pre_join"] = before - int(mask.sum())
        logger.info(
            f"  Step 5 (pre-join weeks): dropped {counts['dropped_pre_join']:,}"
        )
    else:
        counts["dropped_pre_join"] = 0

    df_filtered = df[mask].copy()
    counts["final_rows"] = len(df_filtered)
    counts["final_traders"] = int(df_filtered["trader_id"].nunique())

    logger.info(
        f"Sample construction: {counts['total']:,} → "
        f"{counts['final_rows']:,} "
        f"({counts['final_traders']:,} traders)"
    )

    return df_filtered, counts


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
    # head(1), not first(): first() skips NaN and would mix values across rows
    earliest = df.groupby(group_col).head(1)[[group_col] + existing_conf]

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


def outcome_descriptives(
    df: pd.DataFrame,
    treatment_col: str = TREATMENT,
    outcome_col: str = OUTCOME,
) -> Dict[str, Any]:
    """Facts needed to interpret the estimand (sparsity, identity, raw P&L per trade)."""
    active = df["total_trxn"] > 0
    d = {
        "n_rows": int(len(df)),
        "share_weeks_inactive": float((~active).mean()),
        "share_weeks_zero_gain": float((df["total_gain"] == 0).mean()),
        "outcome_sd_winsorized": float(df[outcome_col].std()),
        "outcome_sd_raw": float(df["total_gain"].std()),
        "treatment_mean": float(df[treatment_col].mean()),
        "treatment_sd": float(df[treatment_col].std()),
        "share_traders_never_self_trade": float(
            (df.groupby("trader_id")[treatment_col].max() == 0).mean()
        ),
    }
    if {"total_self_gain", "total_copied_gain"} <= set(df.columns):
        diff = df["total_gain"] - df["total_self_gain"] - df["total_copied_gain"]
        d["identity_total_eq_self_plus_copied_max_abs_diff"] = float(diff.abs().max())
        n_self = df[treatment_col].sum()
        d["raw_self_gain_per_self_trade"] = (
            float(df["total_self_gain"].sum() / n_self) if n_self > 0 else None
        )
    return d


def period_activity(df: pd.DataFrame, treatment_col: str = TREATMENT) -> pd.DataFrame:
    """
    One row per period: calendar dates, trader-weeks, share active, weeks with
    a self-trade, self-trades, raw self-trade P&L per self-trade, and the share
    of self-trade weeks with exactly zero self-trade gain (a check that gains
    are booked in the week of trading).
    """
    d = df.copy()
    d["_active"] = d["total_trxn"] > 0
    d["_selfweek"] = d[treatment_col] > 0
    g = d.groupby("period")
    out = pd.DataFrame({
        "start_date": pd.to_datetime(g["min_period_date"].min()).dt.date.astype(str),
        "end_date": pd.to_datetime(g["max_period_date"].max()).dt.date.astype(str),
        "trader_weeks": g.size(),
        "share_active": g["_active"].mean(),
        "self_trade_weeks": g["_selfweek"].sum().astype(int),
        "self_trades": g[treatment_col].sum().astype(int),
        "raw_self_gain_per_self_trade": g["total_self_gain"].sum() / g[treatment_col].sum(),
    })
    sw = d[d["_selfweek"]]
    out["share_self_weeks_zero_self_gain"] = (sw["total_self_gain"] == 0).groupby(sw["period"]).mean()
    return out.reset_index()


def activity_descriptives(
    df: pd.DataFrame,
    treatment_col: str = TREATMENT,
    window_start: Optional[pd.Timestamp] = None,
) -> Dict[str, Any]:
    """
    Activity shares, trades per self-trade week, and join-date facts.
    `window_start` is the first day of the observation window (default: the
    earliest week in `df`).
    """
    n = len(df)
    self_weeks = df[treatment_col] > 0
    no_trades = df["total_trxn"] == 0
    vals = df.loc[self_weeks, treatment_col]
    out = {
        "share_no_trades": float(no_trades.mean()),
        "share_copy_only": float((~no_trades & ~self_weeks).mean()),
        "share_self_trade_weeks": float(self_weeks.mean()),
        "n_self_trade_weeks": int(self_weeks.sum()),
        "self_trades_per_self_week_median": float(vals.median()),
        "self_trades_per_self_week_mean": float(vals.mean()),
        "self_trades_per_self_week_max": int(vals.max()),
        "share_self_weeks_single_trade": float((vals == 1).mean()),
        "n_rows": n,
    }
    if "joining_date" in df.columns:
        join = pd.to_datetime(df.groupby("trader_id")["joining_date"].first(), errors="coerce")
        start = (pd.Timestamp(window_start) if window_start is not None
                 else pd.to_datetime(df["min_period_date"]).min())
        out.update({
            "window_start": str(start.date()),
            "share_traders_joined_before_window": float((join < start).mean()),
            "earliest_join_date": str(join.min().date()),
            "max_tenure_weeks": float(df["tenure_weeks"].max()) if "tenure_weeks" in df else None,
        })
    return out
