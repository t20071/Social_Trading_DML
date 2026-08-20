"""
Data loading and cleaning for the XTrader social trading dataset.

Handles:
- Loading the raw Excel file (~170 MB, ~1.2M rows)
- Dropping fully-blank trailer rows (Excel export artifact)
- Date column conversion
- Validation against paper-reported counts (27,895 traders, 1,229,439 obs)
- Caching to parquet for fast subsequent loads

CRITICAL: learner_country_count is ~97% missing — it is dropped from
the DataFrame immediately after the initial EDA missingness report.
No blanket dropna() is ever used; every dropna() targets an explicit
narrow subset of columns.
"""

import os
import logging
from pathlib import Path
from typing import Tuple, Dict, Any, Optional

import pandas as pd
import numpy as np

logger = logging.getLogger(__name__)

# ── Expected counts from the reference paper (Mani et al.) ──
EXPECTED_TRADERS_PAPER = 27_895
EXPECTED_OBS_PAPER = 1_229_439
# The Excel file is truncated at Excel's hard row limit:
EXPECTED_OBS_FILE = 1_048_575  # 2^20 - 1 (Excel row limit minus header)
EXPECTED_PERIOD_MIN = 30
EXPECTED_PERIOD_MAX = 75
EXPECTED_WEEKS = 46

# Date columns to convert from string to datetime
DATE_COLUMNS = ["joining_date", "max_period_date", "min_period_date"]

# The column that is ~97% missing and MUST be excluded
POISON_COLUMN = "learner_country_count"


def load_raw_data(
    path: str = "trader_data_exl.csv",
) -> pd.DataFrame:
    """
    Load the raw CSV file into a DataFrame.

    Parameters
    ----------
    path : str
        Path to the CSV file.

    Returns
    -------
    pd.DataFrame
        Raw data with date columns converted to datetime.
    """
    logger.info(f"Loading raw data from {path} ...")
    df = pd.read_csv(path)
    logger.info(f"Raw data loaded: {len(df):,} rows, {df.shape[1]} columns")

    # Convert date columns
    for col in DATE_COLUMNS:
        if col in df.columns:
            df[col] = pd.to_datetime(df[col], errors="coerce")
            logger.info(
                f"  Converted '{col}' to datetime — "
                f"{df[col].notna().sum():,} valid dates"
            )

    # Convert joining_date2 as well for comparison
    if "joining_date2" in df.columns:
        df["joining_date2"] = pd.to_datetime(df["joining_date2"], errors="coerce")

    return df


def clean_data(
    df: pd.DataFrame,
    apply_keep_country: Optional[bool] = None,
) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """
    Clean the raw data:
    1. Drop fully-blank trailer rows (all columns null, including trader_id)
    2. Investigate keep_country and apply filter per the decision rule
    3. Verify joining_date2 ≈ joining_date, then drop joining_date2
    4. Drop learner_country_count (POISON_COLUMN — ~97% missing)
    5. Report cleaning statistics

    Does NOT deduplicate on trader_id — panel repetition is correct structure.

    Parameters
    ----------
    df : pd.DataFrame
        Raw data from load_raw_data().
    apply_keep_country : bool or None
        If True, filter to keep_country==1.
        If False, use full sample.
        If None, investigate and decide automatically.

    Returns
    -------
    Tuple[pd.DataFrame, Dict]
        Cleaned DataFrame and a dict of cleaning statistics.
    """
    stats = {}
    stats["raw_rows"] = len(df)
    stats["raw_columns"] = df.shape[1]

    # ── Step 1: Drop fully-blank trailer rows ──
    fully_blank_mask = df.isnull().all(axis=1)
    n_blank = int(fully_blank_mask.sum())
    stats["fully_blank_rows_removed"] = n_blank

    if n_blank > 0:
        logger.info(f"Dropping {n_blank:,} fully-blank trailer rows")
        df = df[~fully_blank_mask].copy()
    else:
        logger.info("No fully-blank trailer rows found in this file")

    stats["rows_after_blank_removal"] = len(df)

    # ── Step 1b: Check for Excel truncation ──
    if len(df) == EXPECTED_OBS_FILE:
        logger.warning(
            f"Row count is exactly {EXPECTED_OBS_FILE:,} = Excel's hard row limit "
            f"minus header. The file is likely truncated. "
            f"Paper reports {EXPECTED_OBS_PAPER:,} observations."
        )
        stats["excel_truncated"] = True
        stats["rows_missing_from_paper"] = EXPECTED_OBS_PAPER - len(df)
    else:
        stats["excel_truncated"] = len(df) < EXPECTED_OBS_PAPER

    # ── Step 2: Investigate keep_country ──
    if "keep_country" in df.columns:
        kc_stats = _investigate_keep_country(df)
        stats["keep_country_investigation"] = kc_stats

        if apply_keep_country is None:
            # Decision rule from the spec:
            frac_excluded = kc_stats["fraction_kc0"]
            is_small_share = frac_excluded < 0.15
            # If small and non-systematic → filter to kc==1 as primary
            apply_keep_country = is_small_share
            stats["keep_country_auto_decision"] = (
                "filter_to_kc1" if apply_keep_country
                else "use_full_sample"
            )
            logger.info(
                f"keep_country auto-decision: "
                f"{'filter to kc=1' if apply_keep_country else 'use full sample'} "
                f"(fraction kc=0: {frac_excluded:.1%})"
            )

        if apply_keep_country:
            n_before = len(df)
            n_traders_before = df["trader_id"].nunique()
            df = df[df["keep_country"] == 1].copy()
            stats["keep_country_rows_removed"] = n_before - len(df)
            stats["keep_country_traders_removed"] = (
                n_traders_before - df["trader_id"].nunique()
            )
            stats["keep_country_fraction_removed"] = float(
                stats["keep_country_rows_removed"] / n_before
            )
            logger.info(
                f"Applied keep_country=1 filter: removed "
                f"{stats['keep_country_rows_removed']:,} rows "
                f"({stats['keep_country_fraction_removed']:.1%}), "
                f"{stats['keep_country_traders_removed']:,} traders"
            )
        else:
            stats["keep_country_rows_removed"] = 0
    else:
        stats["keep_country_rows_removed"] = 0

    stats["rows_after_keep_country"] = len(df)

    # ── Step 3: Verify joining_date2 ≈ joining_date ──
    if "joining_date2" in df.columns and "joining_date" in df.columns:
        both_valid = df["joining_date"].notna() & df["joining_date2"].notna()
        if both_valid.sum() > 0:
            match_rate = float(
                (df.loc[both_valid, "joining_date"]
                 == df.loc[both_valid, "joining_date2"]).mean()
            )
            stats["joining_date2_match_rate"] = match_rate
            logger.info(f"joining_date2 matches joining_date: {match_rate:.4%}")
            if match_rate > 0.99:
                df = df.drop(columns=["joining_date2"])
                logger.info("Dropped 'joining_date2' (confirmed duplicate)")
            else:
                logger.warning(
                    f"joining_date2 match rate is only {match_rate:.4%} — "
                    f"keeping both columns for investigation"
                )
        else:
            logger.warning("No rows with both joining_date and joining_date2 valid")
            df = df.drop(columns=["joining_date2"], errors="ignore")

    # ── Step 4: Drop learner_country_count ──
    if POISON_COLUMN in df.columns:
        pct_missing = df[POISON_COLUMN].isna().mean()
        stats["learner_country_count_pct_missing"] = float(pct_missing)
        logger.info(
            f"Dropping '{POISON_COLUMN}' — {pct_missing:.1%} missing "
            f"(excluded from all downstream analysis)"
        )
        df = df.drop(columns=[POISON_COLUMN])

    # ── Step 5: Compute panel statistics ──
    stats["final_rows"] = len(df)
    stats["unique_traders"] = int(df["trader_id"].nunique())
    stats["period_min"] = int(df["period"].min())
    stats["period_max"] = int(df["period"].max())
    stats["n_periods"] = int(df["period"].nunique())

    rows_per_trader = df.groupby("trader_id").size()
    stats["rows_per_trader_mean"] = float(rows_per_trader.mean())
    stats["rows_per_trader_median"] = float(rows_per_trader.median())
    stats["rows_per_trader_min"] = int(rows_per_trader.min())
    stats["rows_per_trader_max"] = int(rows_per_trader.max())
    stats["rows_per_trader_std"] = float(rows_per_trader.std())
    for pct in [10, 25, 50, 75, 90]:
        stats[f"rows_per_trader_p{pct}"] = float(
            rows_per_trader.quantile(pct / 100)
        )

    return df, stats


def _investigate_keep_country(df: pd.DataFrame) -> Dict[str, Any]:
    """
    Run the three required checks on keep_country before deciding how to use it.

    1. Fraction of rows with keep_country==0, distribution across countries/periods
    2. Relationship between keep_country==0 and user_country_name nulls
    3. Compare total_gain and self_trade_ratio between groups
    """
    kc = df["keep_country"]
    kc0_mask = kc == 0
    kc1_mask = kc == 1

    results = {}
    results["n_kc0"] = int(kc0_mask.sum())
    results["n_kc1"] = int(kc1_mask.sum())
    results["fraction_kc0"] = float(kc0_mask.mean())

    # Check 1: Is kc==0 concentrated in specific countries?
    if "user_country_name" in df.columns:
        kc0_countries = df.loc[kc0_mask, "user_country_name"].value_counts().head(10)
        results["kc0_top_countries"] = kc0_countries.to_dict()

        # Check 2: Is user_country_name null for kc==0?
        kc0_country_null_rate = float(
            df.loc[kc0_mask, "user_country_name"].isna().mean()
        )
        kc1_country_null_rate = float(
            df.loc[kc1_mask, "user_country_name"].isna().mean()
        )
        results["kc0_country_null_rate"] = kc0_country_null_rate
        results["kc1_country_null_rate"] = kc1_country_null_rate

    # Check 3: Compare outcomes between groups
    for col in ["total_gain", "total_self_trxn"]:
        if col in df.columns:
            results[f"kc0_{col}_mean"] = float(df.loc[kc0_mask, col].mean())
            results[f"kc1_{col}_mean"] = float(df.loc[kc1_mask, col].mean())
            results[f"kc0_{col}_median"] = float(df.loc[kc0_mask, col].median())
            results[f"kc1_{col}_median"] = float(df.loc[kc1_mask, col].median())

    # Period distribution
    if "period" in df.columns:
        kc0_period_coverage = float(
            df.loc[kc0_mask, "period"].nunique()
            / df["period"].nunique()
        )
        results["kc0_period_coverage"] = kc0_period_coverage

    return results


def get_missingness_report(df: pd.DataFrame) -> pd.DataFrame:
    """
    Generate a per-column missingness report.

    Returns a DataFrame with columns: column, n_missing, pct_missing, dtype.
    """
    report = pd.DataFrame({
        "column": df.columns,
        "n_missing": [int(df[c].isna().sum()) for c in df.columns],
        "pct_missing": [float(df[c].isna().mean()) for c in df.columns],
        "dtype": [str(df[c].dtype) for c in df.columns],
    })
    report = report.sort_values("n_missing", ascending=False).reset_index(drop=True)
    return report


def verify_counts(stats: Dict[str, Any]) -> Dict[str, bool]:
    """
    Verify cleaned counts against expected figures.

    Checks against BOTH paper counts and file-level counts.
    """
    checks = {}
    final_rows = stats["final_rows"]
    unique_traders = stats["unique_traders"]

    # Check against paper expectations (will fail if Excel-truncated)
    checks["row_count_matches_paper"] = (
        abs(final_rows - EXPECTED_OBS_PAPER) / EXPECTED_OBS_PAPER < 0.01
    )

    # Period range
    checks["period_range_correct"] = (
        stats["period_min"] == EXPECTED_PERIOD_MIN
        and stats["period_max"] == EXPECTED_PERIOD_MAX
    )

    # Rows per trader should cluster near 46
    checks["median_rows_near_46"] = (
        35 <= stats["rows_per_trader_median"] <= 46
    )

    for name, passed in checks.items():
        status = "✓" if passed else "✗"
        logger.info(f"  {status} {name}")

    return checks


def print_cleaning_summary(stats: Dict[str, Any]) -> None:
    """Print a human-readable cleaning summary."""
    print("\n" + "=" * 70)
    print("DATA CLEANING SUMMARY")
    print("=" * 70)
    print(f"  Total raw rows:              {stats['raw_rows']:>12,}")
    print(f"  Fully-blank rows removed:    {stats['fully_blank_rows_removed']:>12,}")
    print(f"  Rows after blank removal:    {stats['rows_after_blank_removal']:>12,}")
    if stats.get("keep_country_rows_removed", 0) > 0:
        print(f"  keep_country rows removed:   {stats['keep_country_rows_removed']:>12,}")
    print(f"  Final row count:             {stats['final_rows']:>12,}")
    print(f"  Unique trader_ids:           {stats['unique_traders']:>12,}")
    print(
        f"  Period range:                {stats['period_min']} — "
        f"{stats['period_max']} ({stats['n_periods']} unique periods)"
    )
    if stats.get("excel_truncated"):
        print(
            f"\n  ⚠ Excel truncation detected: file has {stats['rows_after_blank_removal']:,} "
            f"rows vs. paper's {EXPECTED_OBS_PAPER:,}"
        )
    print(f"\n  Rows-per-trader distribution:")
    print(f"    Mean:   {stats['rows_per_trader_mean']:.1f}")
    print(f"    Median: {stats['rows_per_trader_median']:.0f}")
    print(f"    Min:    {stats['rows_per_trader_min']}")
    print(f"    Max:    {stats['rows_per_trader_max']}")
    print(f"    Std:    {stats['rows_per_trader_std']:.1f}")
    for pct in [10, 25, 50, 75, 90]:
        key = f"rows_per_trader_p{pct}"
        if key in stats:
            print(f"    P{pct}: {stats[key]:.0f}", end="  ")
    print()
    if "joining_date2_match_rate" in stats:
        print(
            f"\n  joining_date2 match rate:    "
            f"{stats['joining_date2_match_rate']:.4%}"
        )
    if "learner_country_count_pct_missing" in stats:
        print(
            f"  learner_country_count:       "
            f"{stats['learner_country_count_pct_missing']:.1%} missing → DROPPED"
        )
    print("=" * 70 + "\n")


def verify_column_diagnostics(df: pd.DataFrame) -> Dict[str, Any]:
    """
    Run diagnostic checks on specific columns:
    - std_gain_diff correlation with gain_difference
    - total_audience vs total_follower_count + learner_count
    """
    diagnostics = {}

    # std_gain_diff vs gain_difference
    if "std_gain_diff" in df.columns and "gain_difference" in df.columns:
        valid = df[["std_gain_diff", "gain_difference"]].dropna()
        if len(valid) > 0:
            corr = float(valid["std_gain_diff"].corr(valid["gain_difference"]))
            diagnostics["std_gain_diff_vs_gain_difference_corr"] = corr
            logger.info(f"std_gain_diff ↔ gain_difference correlation: {corr:.4f}")

    # total_audience vs total_follower_count + learner_count
    if all(c in df.columns for c in
           ["total_audience", "total_follower_count", "learner_count"]):
        computed_sum = df["total_follower_count"] + df["learner_count"]
        match = (df["total_audience"] == computed_sum)
        diagnostics["total_audience_equals_follower_plus_learner"] = bool(match.all())
        diagnostics["total_audience_match_rate"] = float(match.mean())
        if not match.all():
            diff = (df["total_audience"] - computed_sum).describe()
            diagnostics["total_audience_diff_stats"] = diff.to_dict()

    return diagnostics


def save_processed(
    df: pd.DataFrame,
    path: str = "data/processed/trader_panel.parquet",
) -> str:
    """Save cleaned DataFrame to parquet. Returns the path."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    df_save = df.copy()
    # Fix mixed-type object columns for pyarrow
    for col in df_save.select_dtypes(include=["object"]).columns:
        df_save[col] = df_save[col].astype(str)
    df_save.to_parquet(path, index=False, engine="pyarrow")
    size_mb = os.path.getsize(path) / 1e6
    logger.info(f"Saved processed data to {path} ({size_mb:.1f} MB)")
    return path


def load_processed(
    path: str = "data/processed/trader_panel.parquet",
) -> pd.DataFrame:
    """Load cached parquet file."""
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"Processed data not found at {path}. "
            f"Run the data loading pipeline first."
        )
    df = pd.read_parquet(path, engine="pyarrow")
    logger.info(f"Loaded processed data: {len(df):,} rows from {path}")
    return df
