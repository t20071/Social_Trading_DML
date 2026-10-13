"""
Tests for data cleaning and feature engineering.

- Lag correctness and row-count preservation
- Tenure computation
- dropna audit (no blanket dropna anywhere in the package)
- learner_country_count exclusion guard
- Sample construction keeps the panel structure
"""

import os

import numpy as np
import pandas as pd
import pytest

from tests.synthetic import make_test_panel


class TestLags:
    """Test that lagged confounders are computed correctly."""

    def test_lag_creates_columns(self):
        from social_trading_dml.data.features import compute_lags
        df = make_test_panel()
        df_lagged = compute_lags(df, lag_columns=["learner_count"])
        assert "learner_count_l1" in df_lagged.columns

    def test_lag_values_correct(self):
        """For each trader, _l1 at period t should equal raw at period t-1."""
        from social_trading_dml.data.features import compute_lags
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
        from social_trading_dml.data.features import compute_lags
        df = make_test_panel()
        df_lagged = compute_lags(df, lag_columns=["learner_count"])

        # head(1), not first(): first() skips NaN and returns the 2nd row's lag
        first_obs = df_lagged.sort_values(["trader_id", "period"]).groupby(
            "trader_id").head(1)
        assert first_obs["learner_count_l1"].isna().all()

    def test_row_count_preserved(self):
        """Lagging should not change the number of rows."""
        from social_trading_dml.data.features import compute_lags
        df = make_test_panel()
        n_before = len(df)
        df_lagged = compute_lags(df)
        assert len(df_lagged) == n_before


class TestTenure:
    """Test tenure computation."""

    def test_tenure_nonnegative(self):
        from social_trading_dml.data.features import compute_tenure
        df = make_test_panel()
        df_t = compute_tenure(df)
        assert (df_t["tenure_weeks"] >= 0).all()

    def test_tenure_increases_with_period(self):
        """For a given trader, tenure should increase with period."""
        from social_trading_dml.data.features import compute_tenure
        df = make_test_panel(n_traders=1, n_periods=10)
        df_t = compute_tenure(df).sort_values("period")
        diffs = df_t["tenure_weeks"].diff().dropna()
        assert (diffs >= 0).all()


class TestDropnaAudit:
    """Ensure no blanket dropna() anywhere in the codebase."""

    def test_no_blanket_dropna_in_source(self):
        """
        Scan src/ for DataFrame.dropna() calls without subset=.
        Any DataFrame dropna() must specify subset=... to avoid the
        learner_country_count catastrophe.

        Parses the source with ast so docstrings are ignored. A no-argument
        dropna() is allowed only on a Series (receiver is a column selection
        df[...], df.loc[...], or a Series method chain on one).
        """
        import ast
        src_dir = os.path.join(os.path.dirname(__file__), "..", "src", "social_trading_dml")
        violations = []

        def is_series_expr(node):
            # df["col"], df.loc[mask, "col"], df[...].something(...)
            while isinstance(node, (ast.Call, ast.Attribute)):
                node = node.func if isinstance(node, ast.Call) else node.value
            return isinstance(node, ast.Subscript)

        py_files = [os.path.join(d, f) for d, _, fs in os.walk(src_dir)
                    for f in fs if f.endswith(".py")]
        assert py_files, "no source files found"
        for fpath in py_files:
            fname = os.path.relpath(fpath, src_dir)
            with open(fpath, encoding="utf-8") as f:
                tree = ast.parse(f.read())
            for node in ast.walk(tree):
                if (isinstance(node, ast.Call)
                        and isinstance(node.func, ast.Attribute)
                        and node.func.attr == "dropna"):
                    kw = {k.arg for k in node.keywords}
                    if "subset" in kw or node.args:
                        continue
                    if is_series_expr(node.func.value):
                        continue
                    violations.append(f"{fname}:{node.lineno}")

        assert len(violations) == 0, (
            f"Found blanket dropna() calls (no subset=):\n"
            + "\n".join(violations)
        )

    def test_learner_country_count_not_in_confounders(self):
        """learner_country_count must never appear in any confounder list."""
        from social_trading_dml.data.features import CORE_LAG_COLUMNS, get_excluded_columns
        assert "learner_country_count" not in CORE_LAG_COLUMNS
        assert "learner_country_count" in get_excluded_columns()


class TestSampleConstruction:
    """Test the analysis sample construction logic."""

    def test_sample_construction_preserves_panel(self):
        """Sample construction should not accidentally deduplicate trader_ids."""
        from social_trading_dml.data.features import compute_lags
        from social_trading_dml.data.variables import compute_treatment, get_analysis_sample

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
