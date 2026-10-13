"""
Feature engineering for the causal ML analysis.

Implements:
- Lagged confounders (t-1) with mandatory audit verification
- Tenure calculation (weeks since joining)
- Country bucketing (top ~15 + "Other")
- total_trxn lagging for activity control
- Column exclusion list per the variable decision table

CRITICAL: The lag operation uses groupby(trader_id).shift(1) within
a pre-sorted (trader_id, period) panel. The print_lag_audit_table()
function MUST be called before any model fit to visually verify that
each _l1 value equals the raw value from the previous period for the
same trader.
"""

import logging
from typing import List, Optional, Dict, Tuple

import pandas as pd
import numpy as np

logger = logging.getLogger(__name__)

# Core confounders to lag (from period t to t-1)
# NOTE: total_trxn is added as an explicit control per the spec note.
# total_self_trxn_l1 (lagged treatment) controls for trading-habit persistence.
# The lagged outcome is deliberately NOT included: under trader fixed effects
# it induces Nickell (1981) bias; gain_difference_l1 carries the prior-week
# gain feedback used in Mani et al. (Eq. 1).
CORE_LAG_COLUMNS = [
    "learner_count",
    "total_follower_count",
    "guru_count",
    "followed_count",
    "gain_difference",
    "greater_than_ten",
    "total_trxn",
    "total_self_trxn",
]

# Minimum weeks a trader must be on the platform within the 46-week window
# for the "established traders" robustness sample (Mani et al., p. 21).
MIN_WEEKS_PRESENT = 23

# Audience link counts with whole-week recording gaps in the raw data: in some
# weeks (almost) every trader has 0, e.g. copy links (learner/guru) in weeks
# 38-40, 44, 50 and follow links (follower/followed) in weeks 32-34, 42.
AUDIENCE_COLUMNS = [
    "learner_count",
    "guru_count",
    "total_follower_count",
    "followed_count",
]

# Pre-computed log columns to EXCLUDE (never use — see decision table)
EXCLUDED_LOG_COLUMNS = [
    "log_follower_count",
    "log_followed_count",
    "log_guru_count",
    "log_learner_count",
    "log_total_audience",
    "log_foreign_followers",
    "log_domestic_followers",
    "log_foreign_learners",
    "log_domestic_learners",
    "log_active_followed_trader_count",
]

# Identifier columns excluded from modeling
EXCLUDED_IDENTIFIER_COLUMNS = [
    "trader_name",
    "trader",
]

# Outcome components — NEVER enter models as confounders
EXCLUDED_OUTCOME_COMPONENTS = [
    "total_self_gain",
    "total_copied_gain",
    "total_copied_trxn",
]

# Redundant / near-duplicate / high-missingness columns
EXCLUDED_DUPLICATES = [
    "std_gain_diff",           # near-duplicate of gain_difference
    "learner_country_count",   # ~97% missing — should already be dropped
    "follower_country_count",
    "foreign_country_follower_count",
    "foreign_country_learner_count",
    "same_country_follower_count",
    "same_country_learner_count",
]


def get_excluded_columns() -> List[str]:
    """
    Return the full list of columns excluded from modeling.

    These are excluded per the variable decision table:
    - Pre-computed log transforms
    - Identifier columns (trader_name, trader)
    - Outcome components
    - Near-duplicates (std_gain_diff)
    - joining_date2 (verified duplicate, dropped in cleaning)
    """
    return (
        EXCLUDED_LOG_COLUMNS
        + EXCLUDED_IDENTIFIER_COLUMNS
        + EXCLUDED_OUTCOME_COMPONENTS
        + EXCLUDED_DUPLICATES
        + ["joining_date2"]
    )


def repair_audience_gaps(
    df: pd.DataFrame,
    cols: Optional[List[str]] = None,
    threshold: float = 0.5,
) -> Tuple[pd.DataFrame, Dict[str, List[int]]]:
    """
    Treat whole-week recording gaps in audience counts as missing, not zero.

    A week is a gap for a column when its number of non-zero values is below
    `threshold` x the median weekly number of non-zero values. In gap weeks the
    ZERO values are treated as missing and filled with the trader's previous
    recorded value (next value if there is none); non-zero values recorded in a
    partially recorded week are kept. 'audience_imputed' flags filled rows.

    Returns the repaired DataFrame and {column: [gap periods]}.
    """
    if cols is None:
        cols = AUDIENCE_COLUMNS
    df = df.sort_values(["trader_id", "period"]).copy()
    df["audience_imputed"] = 0
    gaps = {}
    for col in cols:
        if col not in df.columns:
            continue
        nonzero = (df[col] != 0).groupby(df["period"]).sum()
        gap_periods = sorted(nonzero.index[nonzero < threshold * nonzero.median()].tolist())
        gaps[col] = [int(p) for p in gap_periods]
        if not gap_periods:
            continue
        to_fill = df["period"].isin(gap_periods) & (df[col] == 0)
        df[col] = df[col].astype(np.float64).where(~to_fill)
        df[col] = df.groupby("trader_id")[col].ffill()
        df[col] = df.groupby("trader_id")[col].bfill()
        df.loc[to_fill, "audience_imputed"] = 1
        logger.info(f"  {col}: recording-gap weeks {gaps[col]} filled from adjacent weeks")
    return df, gaps


def compute_lags(
    df: pd.DataFrame,
    lag_columns: Optional[List[str]] = None,
    groupby: str = "trader_id",
    sort_col: str = "period",
    lag: int = 1,
) -> pd.DataFrame:
    """
    Create lagged versions of confounder columns within each trader.

    For each column in `lag_columns`, creates `{col}_l1` = value at
    period t-1 for the same trader. Rows with no prior period
    (first observation) get NaN.

    Parameters
    ----------
    df : pd.DataFrame
        Panel data sorted by (trader_id, period).
    lag_columns : list of str, optional
        Columns to lag. Defaults to CORE_LAG_COLUMNS.
    groupby : str
        Panel ID column (default: 'trader_id').
    sort_col : str
        Time column to sort within group (default: 'period').
    lag : int
        Number of periods to lag (default: 1).

    Returns
    -------
    pd.DataFrame
        DataFrame with new `{col}_l1` columns added.
    """
    if lag_columns is None:
        lag_columns = CORE_LAG_COLUMNS

    # Ensure sorted
    df = df.sort_values([groupby, sort_col]).copy()

    for col in lag_columns:
        if col not in df.columns:
            logger.warning(f"Column '{col}' not found in DataFrame — skipping lag")
            continue
        lag_col_name = f"{col}_l{lag}"
        df[lag_col_name] = df.groupby(groupby)[col].shift(lag)
        n_null = int(df[lag_col_name].isna().sum())
        logger.info(
            f"  Created '{lag_col_name}': {n_null:,} NaN values "
            f"({n_null / len(df):.1%} of rows)"
        )

    return df


def print_lag_audit_table(
    df: pd.DataFrame,
    lag_columns: Optional[List[str]] = None,
    sample_trader_ids: Optional[List[int]] = None,
    n_sample: int = 4,
    n_periods: int = 5,
) -> pd.DataFrame:
    """
    MANDATORY verification: print a side-by-side audit table showing raw and
    lagged values for sample traders, so the lag is visually confirmed.

    For each sample trader, shows `period`, the raw confounder, and its _l1
    version side by side. The _l1 value at period t should equal the raw
    value at period t-1 for the same trader.

    Returns
    -------
    pd.DataFrame
        The audit table (also printed to stdout).
    """
    if lag_columns is None:
        lag_columns = CORE_LAG_COLUMNS

    if sample_trader_ids is None:
        # Pick traders who have enough observations for a meaningful audit
        trader_counts = df.groupby("trader_id").size()
        eligible = trader_counts[trader_counts >= n_periods + 1].index
        if len(eligible) < n_sample:
            eligible = trader_counts.index
        rng = np.random.RandomState(42)
        sample_trader_ids = rng.choice(
            eligible, size=min(n_sample, len(eligible)), replace=False
        )

    audit_rows = []
    for tid in sample_trader_ids:
        trader_df = (
            df[df["trader_id"] == tid]
            .sort_values("period")
            .head(n_periods)
        )
        audit_rows.append(trader_df)

    audit_df = pd.concat(audit_rows)

    # Select columns for display: show first 3 lag pairs for readability
    display_cols = ["trader_id", "period"]
    for col in lag_columns[:3]:
        if col in audit_df.columns:
            display_cols.append(col)
        lag_col = f"{col}_l1"
        if lag_col in audit_df.columns:
            display_cols.append(lag_col)

    display_cols = [c for c in display_cols if c in audit_df.columns]
    audit_display = audit_df[display_cols].copy()

    print("\n" + "=" * 100)
    print("LAG VERIFICATION AUDIT TABLE")
    print(
        "Verify: each _l1 value should equal the raw value "
        "from the PREVIOUS period for the same trader"
    )
    print("=" * 100)
    print(audit_display.to_string(index=False))
    print("=" * 100 + "\n")

    return audit_display


def compute_tenure(df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute tenure = weeks since joining_date as of the period's
    min_period_date.

    Tenure is not lagged — it is pre-determined at any point in time.
    Rows with missing joining_date get NaN tenure (handled separately
    in sample construction).

    Also adds:
    - 'pre_join': True for weeks that END before the trader joined
      (max_period_date < joining_date; falls back to tenure < 0). A trader
      joining mid-week keeps that week. Pre-join rows are excluded from the
      analysis sample.
    - 'weeks_present': per-trader count of post-join weeks in the window,
      used for the >= MIN_WEEKS_PRESENT robustness sample.

    Parameters
    ----------
    df : pd.DataFrame
        Must contain 'joining_date' (datetime) and 'min_period_date' (datetime).

    Returns
    -------
    pd.DataFrame
        DataFrame with 'tenure_weeks' column added.
    """
    df = df.copy()

    if "joining_date" not in df.columns or "min_period_date" not in df.columns:
        raise ValueError("Need 'joining_date' and 'min_period_date' columns")

    # Ensure datetime
    df["joining_date"] = pd.to_datetime(df["joining_date"], errors="coerce")
    df["min_period_date"] = pd.to_datetime(df["min_period_date"], errors="coerce")

    df["tenure_weeks"] = (
        (df["min_period_date"] - df["joining_date"]).dt.days / 7.0
    )

    # Weeks that end before joining are pre-join; negative tenure (a week that
    # starts before joining) is clamped to 0
    if "max_period_date" in df.columns:
        max_date = pd.to_datetime(df["max_period_date"], errors="coerce")
        df["pre_join"] = (max_date < df["joining_date"]).fillna(False)
    else:
        df["pre_join"] = df["tenure_weeks"] < 0
    n_pre = int(df["pre_join"].sum())
    n_negative = int((df["tenure_weeks"] < 0).sum())
    if n_negative > 0:
        logger.warning(
            f"{n_negative:,} rows have negative tenure; {n_pre:,} of them end "
            f"before joining (flagged pre_join); tenure clamped to 0"
        )
        df["tenure_weeks"] = df["tenure_weeks"].clip(lower=0)

    present = df["tenure_weeks"].notna() & ~df["pre_join"]
    df["weeks_present"] = present.groupby(df["trader_id"]).transform("sum").astype(int)

    # Report distribution
    valid_tenure = df["tenure_weeks"].dropna()
    logger.info(f"Tenure distribution (valid):\n{valid_tenure.describe()}")
    skew = float(valid_tenure.skew())
    logger.info(f"Tenure skewness: {skew:.2f}")

    return df


def add_pre_window_moderators(df: pd.DataFrame) -> pd.DataFrame:
    """
    Trader-level effect modifiers fixed before the analysis window, so they
    cannot respond to the treatment:

    - join_cohort_weeks: weeks between joining the platform and the start of
      the window (period 30); negative for traders who joined later
    - copiers_start / followers_start: copier and follower audience in the
      first period of the window (period 30)
    """
    df = df.copy()
    first = df.sort_values(["trader_id", "period"]).groupby("trader_id").head(1)
    window_start = pd.to_datetime(df["min_period_date"]).min()
    joined = pd.to_datetime(first["joining_date"], errors="coerce")
    per_trader = pd.DataFrame({
        "trader_id": first["trader_id"].to_numpy(),
        "join_cohort_weeks": ((window_start - joined).dt.days / 7.0).to_numpy(),
        "copiers_start": first["learner_count"].to_numpy(dtype=np.float64),
        "followers_start": first["total_follower_count"].to_numpy(dtype=np.float64),
    })
    df = df.drop(columns=[c for c in per_trader.columns if c != "trader_id" and c in df.columns])
    return df.merge(per_trader, on="trader_id", how="left")


def bucket_country(
    df: pd.DataFrame,
    col: str = "user_country_name",
    top_n: int = 15,
    other_label: str = "Other",
) -> pd.DataFrame:
    """
    Bucket country to top_n most frequent + "Other".

    Creates '{col}_bucketed' as a categorical column.
    Does NOT one-hot encode — that happens at model fit time.
    """
    df = df.copy()

    if col not in df.columns:
        logger.warning(f"Column '{col}' not found — skipping country bucketing")
        return df

    top_countries = df[col].value_counts().head(top_n).index.tolist()
    bucketed_col = f"{col}_bucketed"
    df[bucketed_col] = df[col].where(
        df[col].isin(top_countries), other=other_label
    )
    # Handle NaN: fill with "Unknown" to avoid losing rows
    df[bucketed_col] = df[bucketed_col].fillna("Unknown")
    df[bucketed_col] = df[bucketed_col].astype("category")

    n_raw = df[col].nunique()
    n_bucketed = df[bucketed_col].nunique()
    logger.info(
        f"Country bucketing: {n_raw} → {n_bucketed} categories "
        f"(top {top_n} + '{other_label}' + 'Unknown')"
    )

    return df


def get_country_dummies(
    df: pd.DataFrame,
    col: str = "user_country_name_bucketed",
    drop_first: bool = True,
) -> Tuple[pd.DataFrame, List[str]]:
    """
    One-hot encode the bucketed country column for linear models.

    Returns
    -------
    Tuple[pd.DataFrame, List[str]]
        DataFrame with dummies appended, and list of dummy column names.
    """
    dummies = pd.get_dummies(df[col], prefix="country", drop_first=drop_first)
    # Ensure all columns are numeric (bool → int)
    dummies = dummies.astype(int)
    dummy_cols = dummies.columns.tolist()
    df = pd.concat([df, dummies], axis=1)
    return df, dummy_cols


def build_features(
    df: pd.DataFrame,
    lag_columns: Optional[List[str]] = None,
    top_n_countries: int = 15,
    print_audit: bool = True,
) -> pd.DataFrame:
    """
    Full feature engineering pipeline:
    1. Compute lagged confounders
    2. Print lag audit table (mandatory verification)
    3. Compute tenure
    4. Bucket country

    Parameters
    ----------
    df : pd.DataFrame
        Cleaned panel data.
    lag_columns : list of str, optional
        Columns to lag (default: CORE_LAG_COLUMNS).
    top_n_countries : int
        Number of top countries to keep.
    print_audit : bool
        Whether to print the lag audit table.

    Returns
    -------
    pd.DataFrame
        DataFrame with all engineered features.
    """
    logger.info("Starting feature engineering pipeline ...")

    # Step 0: Audience recording gaps -> missing, filled from adjacent weeks
    logger.info("Step 0: Repairing audience-count recording gaps")
    df, gaps = repair_audience_gaps(df)

    # Step 1: Lags
    logger.info("Step 1: Computing lagged confounders")
    df = compute_lags(df, lag_columns=lag_columns)

    # Step 2: Audit
    if print_audit:
        logger.info("Step 2: Lag verification audit")
        print_lag_audit_table(df, lag_columns=lag_columns)

    # Step 3: Tenure
    logger.info("Step 3: Computing tenure")
    df = compute_tenure(df)

    # Step 4: Country bucketing
    logger.info("Step 4: Bucketing country")
    df = bucket_country(df, top_n=top_n_countries)

    # Step 5: Pre-window effect modifiers (join cohort, starting audience)
    if {"joining_date", "min_period_date", "learner_count", "total_follower_count"} <= set(df.columns):
        logger.info("Step 5: Pre-window moderators")
        df = add_pre_window_moderators(df)

    df.attrs["audience_gap_weeks"] = gaps
    logger.info(
        f"Feature engineering complete: {len(df):,} rows, "
        f"{df.shape[1]} columns"
    )
    return df
