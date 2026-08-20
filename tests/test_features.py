"""
Sanity checks for the feature engineering pipeline.

Tests:
- Row count preservation through feature engineering
- Lag formula correctness
- dropna audit (no blanket dropna)
- learner_country_count exclusion guard
- Tenure computation logic
"""

import os
import sys
import pytest

# Ensure src/ is importable
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
import pandas as pd


def make_test_panel(n_traders=5, n_periods=10):
    """Create a small test panel for unit tests."""
    rows = []
    for tid in range(1, n_traders + 1):
        for period in range(30, 30 + n_periods):
            rows.append({
                "trader_id": tid,
                "period": period,
                "learner_count": np.random.randint(0, 100),
                "total_follower_count": np.random.randint(0, 200),
                "guru_count": np.random.randint(0, 10),
                "followed_count": np.random.randint(0, 50),
                "gain_difference": np.random.randn() * 100,
                "greater_than_ten": np.random.randint(0, 2),
                "total_trxn": np.random.randint(0, 50),
                "total_self_trxn": np.random.randint(0, 30),
                "total_gain": np.random.randn() * 1000,
                "user_country_name": np.random.choice(
                    ["USA", "UK", "Germany", "India", "Unknown"]
                ),
                "joining_date": pd.Timestamp("2012-06-15"),
                "min_period_date": pd.Timestamp("2013-01-01")
                    + pd.Timedelta(weeks=period - 30),
                "max_period_date": pd.Timestamp("2013-01-07")
                    + pd.Timedelta(weeks=period - 30),
                "keep_country": 1,
                "learner_country_count": np.nan,  # ~97% missing in real data
            })
    return pd.DataFrame(rows)


class TestLags:
    """Test that lagged confounders are computed correctly."""

    def test_lag_creates_columns(self):
        from src.features import compute_lags
        df = make_test_panel()
        df_lagged = compute_lags(df, lag_columns=["learner_count"])
        assert "learner_count_l1" in df_lagged.columns

    def test_lag_values_correct(self):
        """For each trader, _l1 at period t should equal raw at period t-1."""
        from src.features import compute_lags
        df = make_test_panel()
        df_lagged = compute_lags(df, lag_columns=["learner_count"])

        for tid in df_lagged["trader_id"].unique():
            trader_df = df_lagged[df_lagged["trader_id"] == tid].sort_values("period")
            raw = trader_df["learner_count"].values
            lagged = trader_df["learner_count_l1"].values
            # First period should be NaN (no prior)
            assert np.isnan(lagged[0])
            # Remaining should match the previous raw value
            np.testing.assert_array_equal(lagged[1:], raw[:-1])

    def test_lag_first_period_is_nan(self):
        """First observation per trader should have NaN lag."""
        from src.features import compute_lags
        df = make_test_panel()
        df_lagged = compute_lags(df, lag_columns=["learner_count"])

        first_obs = df_lagged.groupby("trader_id").first()
        assert first_obs["learner_count_l1"].isna().all()

    def test_row_count_preserved(self):
        """Lagging should not change the number of rows."""
        from src.features import compute_lags
        df = make_test_panel()
        n_before = len(df)
        df_lagged = compute_lags(df)
        assert len(df_lagged) == n_before


class TestTenure:
    """Test tenure computation."""

    def test_tenure_nonnegative(self):
        from src.features import compute_tenure
        df = make_test_panel()
        df_t = compute_tenure(df)
        assert (df_t["tenure_weeks"] >= 0).all()

    def test_tenure_increases_with_period(self):
        """For a given trader, tenure should increase with period."""
        from src.features import compute_tenure
        df = make_test_panel(n_traders=1, n_periods=10)
        df_t = compute_tenure(df).sort_values("period")
        diffs = df_t["tenure_weeks"].diff().dropna()
        assert (diffs >= 0).all()


class TestDropnaAudit:
    """Ensure no blanket dropna() anywhere in the codebase."""

    def test_no_blanket_dropna_in_source(self):
        """
        Scan all .py files in src/ for dropna() calls without subset.
        Any dropna() must specify subset=... to avoid the
        learner_country_count catastrophe.
        """
        import re
        src_dir = os.path.join(os.path.dirname(__file__), "..", "src")
        violations = []

        for fname in os.listdir(src_dir):
            if not fname.endswith(".py"):
                continue
            fpath = os.path.join(src_dir, fname)
            with open(fpath) as f:
                lines = f.readlines()
            for i, line in enumerate(lines, 1):
                # Match .dropna() with no arguments or with only axis=
                if re.search(r"\.dropna\(\s*\)", line):
                    violations.append(f"{fname}:{i}: {line.strip()}")

        assert len(violations) == 0, (
            f"Found blanket dropna() calls (no subset=):\n"
            + "\n".join(violations)
        )

    def test_learner_country_count_not_in_confounders(self):
        """learner_country_count must never appear in any confounder list."""
        from src.features import CORE_LAG_COLUMNS, get_excluded_columns
        assert "learner_country_count" not in CORE_LAG_COLUMNS
        assert "learner_country_count" in get_excluded_columns()


class TestSampleConstruction:
    """Test the analysis sample construction logic."""

    def test_sample_construction_preserves_panel(self):
        """Sample construction should not accidentally deduplicate trader_ids."""
        from src.features import compute_lags
        from src.treatment_outcome import compute_treatment, get_analysis_sample

        df = make_test_panel(n_traders=5, n_periods=10)
        df = compute_lags(df)
        df, _ = compute_treatment(df)
        df["tenure_weeks"] = 10.0  # Mock tenure
        df["total_gain_w"] = df["total_gain"]

        lag_cols = [f"{c}_l1" for c in ["learner_count", "total_follower_count",
                    "guru_count", "followed_count", "gain_difference",
                    "greater_than_ten", "total_trxn"]]

        df_sample, counts = get_analysis_sample(
            df, treatment_col="total_self_trxn",
            confounder_lag_cols=lag_cols,
        )

        # Should have dropped first period per trader (NaN lags) but kept rest
        # Each trader loses 1 period → 5 * (10-1) = 45 rows
        expected = 5 * (10 - 1)
        assert len(df_sample) == expected
        assert df_sample["trader_id"].nunique() == 5


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
