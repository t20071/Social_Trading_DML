"""
Tests for outcome construction, fixed-effects transforms, the model cache,
the estimators and the diagnostics.

The key test simulates a panel with trader effects and week shocks that drive
both treatment and outcome, and checks that the primary two-way FE DML
recovers the true effect while the pooled estimator does not.
"""

import os

import numpy as np
import pandas as pd
import pytest

from tests.synthetic import simulate_panel





class TestDemeaning:
    def test_two_way_means_zero_balanced(self):
        from social_trading_dml.estimation.panel import demean_within_trader
        df = simulate_panel(50, 10)
        out = demean_within_trader(df, ["T", "Y"], time_col="period")
        for c in ["T_demeaned", "Y_demeaned"]:
            assert out.groupby("trader_id")[c].mean().abs().max() < 1e-8
            assert out.groupby("period")[c].mean().abs().max() < 1e-8

    def test_two_way_means_zero_unbalanced(self):
        from social_trading_dml.estimation.panel import demean_within_trader
        df = simulate_panel(50, 10).sample(frac=0.8, random_state=1)
        out = demean_within_trader(df, ["T"], time_col="period")
        assert out.groupby("trader_id")["T_demeaned"].mean().abs().max() < 1e-6
        assert out.groupby("period")["T_demeaned"].mean().abs().max() < 1e-6

    def test_one_way_matches_manual(self):
        from social_trading_dml.estimation.panel import demean_within_trader
        df = simulate_panel(20, 5)
        out = demean_within_trader(df, ["T"])
        manual = df["T"] - df.groupby("trader_id")["T"].transform("mean")
        np.testing.assert_allclose(out["T_demeaned"], manual)

    def test_first_difference_handles_gaps(self):
        from social_trading_dml.estimation.panel import first_difference
        df = pd.DataFrame({"trader_id": [1, 1, 1, 2, 2],
                           "period": [30, 31, 33, 30, 31],
                           "v": [1.0, 3.0, 10.0, 5.0, 4.0]})
        out = first_difference(df, ["v"]).set_index(["trader_id", "period"])["v_fd"]
        assert np.isnan(out[(1, 30)]) and out[(1, 31)] == 2.0
        assert np.isnan(out[(1, 33)])  # gap: 31 -> 33 is not consecutive
        assert out[(2, 31)] == -1.0


class TestOutcome:
    def test_winsorize_uses_nonzero_values(self):
        from social_trading_dml.data.variables import compute_outcome
        rng = np.random.default_rng(0)
        n_active, n_inactive = 1000, 9000
        df = pd.DataFrame({
            "total_trxn": np.r_[np.ones(n_active), np.zeros(n_inactive)],
            "total_gain": np.r_[rng.normal(0, 100, n_active), np.zeros(n_inactive)],
        })
        out = compute_outcome(df)
        active_gain = df.loc[df.total_trxn > 0, "total_gain"]
        assert out["total_gain_w"].max() == pytest.approx(active_gain.quantile(0.99))
        assert out["total_gain_w"].min() == pytest.approx(active_gain.quantile(0.01))
        # Inactive weeks stay exactly zero
        assert (out.loc[df.total_trxn == 0, "total_gain_w"] == 0).all()
        # About 2% of active weeks clipped, not most of them
        clipped = (out["total_gain_w"] != df["total_gain"])[df.total_trxn > 0].mean()
        assert clipped < 0.03


class TestTenureFlags:
    def test_pre_join_and_weeks_present(self):
        from social_trading_dml.data.features import compute_tenure
        dates = pd.date_range("2013-01-01", periods=6, freq="7D")
        df = pd.DataFrame({
            "trader_id": [1] * 6 + [2] * 6,
            "period": list(range(30, 36)) * 2,
            "min_period_date": list(dates) * 2,
            "joining_date": [pd.Timestamp("2012-01-01")] * 6
                            + [pd.Timestamp("2013-01-20")] * 6,
        })
        out = compute_tenure(df)
        assert not out.loc[out.trader_id == 1, "pre_join"].any()
        assert out.loc[out.trader_id == 2, "pre_join"].sum() == 3
        assert (out.loc[out.trader_id == 1, "weeks_present"] == 6).all()
        assert (out.loc[out.trader_id == 2, "weeks_present"] == 3).all()
        assert (out["tenure_weeks"] >= 0).all()

    def test_analysis_sample_drops_pre_join(self):
        from social_trading_dml.data.variables import get_analysis_sample
        df = pd.DataFrame({
            "trader_id": [1, 1, 1], "total_self_trxn": [0, 1, 2],
            "total_gain": [0.0, 1.0, 2.0], "tenure_weeks": [0.0, 0.0, 1.0],
            "pre_join": [True, False, False],
        })
        out, counts = get_analysis_sample(df, confounder_lag_cols=[])
        assert len(out) == 2 and counts["dropped_pre_join"] == 1


class TestCaching:
    def test_fingerprint_changes_with_data_and_params(self):
        from social_trading_dml.estimation.cache import fingerprint
        a = np.arange(10.0)
        b = a.copy()
        b[3] += 1
        assert fingerprint(a, learner="histgbr") == fingerprint(a.copy(), learner="histgbr")
        assert fingerprint(a, learner="histgbr") != fingerprint(b, learner="histgbr")
        assert fingerprint(a, learner="histgbr") != fingerprint(a, learner="lasso")
        assert fingerprint(a[:5]) != fingerprint(a)

    def test_cache_dir_is_absolute(self):
        from social_trading_dml.estimation.cache import CACHE_DIR
        assert os.path.isabs(CACHE_DIR)


class TestEstimators:
    def test_clustered_final_stage_point_estimate(self):
        from social_trading_dml.estimation.dml import clustered_final_stage
        rng = np.random.default_rng(0)
        t = rng.normal(size=500)
        y = -1.5 * t + rng.normal(size=500)
        g = np.repeat(np.arange(50), 10)
        res = clustered_final_stage(y, t, g)
        assert res["ate"] == pytest.approx((t @ y) / (t @ t))
        assert res["n_clusters"] == 50 and res["se_type"] == "cluster(trader)"

    def test_build_fe_arrays_drops_absorbed_controls(self):
        from social_trading_dml.estimation.panel import build_fe_arrays
        df = simulate_panel(30, 6)
        df["const_in_trader"] = df["trader_id"] * 1.0   # absorbed by trader FE
        arrays = build_fe_arrays(df, "T", "Y", ["x", "const_in_trader"])
        assert arrays["W_names"] == ["x"]

    def test_two_way_fe_dml_recovers_effect(self):
        """Trader effects and week shocks confound T->Y; two-way FE DML removes both."""
        from social_trading_dml.estimation.panel import build_fe_arrays
        from social_trading_dml.estimation.dml import fit_linear_dml
        df = simulate_panel(n_traders=300, n_periods=20, theta=-2.0)

        fe = build_fe_arrays(df, "T", "Y", ["x"], fe="twoway")
        _, res = fit_linear_dml(fe["Y"], fe["T"], fe["W"], groups=fe["groups"])
        assert res["ate_ci_lower"] < -2.0 < res["ate_ci_upper"]
        assert abs(res["ate"] + 2.0) < 0.1

        pooled = build_fe_arrays(df, "T", "Y", ["x"], fe="none")
        _, res_p = fit_linear_dml(pooled["Y"], pooled["T"], pooled["W"],
                                  groups=pooled["groups"])
        assert abs(res_p["ate"] + 2.0) > 0.5   # confounded

    def test_gates_detects_heterogeneity(self):
        """GATES machinery: x plays the role of a held-out CATE prediction."""
        from social_trading_dml.estimation.heterogeneity import gates_test
        rng = np.random.default_rng(1)
        n = 8000
        x = rng.normal(size=n)
        t = rng.normal(size=n)
        g = np.repeat(np.arange(n // 10), 10)
        y_het = (1.0 + 2.0 * (x > 0)) * t + rng.normal(size=n)
        y_hom = 1.0 * t + rng.normal(size=n)
        assert gates_test(x, y_het, t, g)["wald_equal_pvalue"] < 1e-6
        assert gates_test(x, y_hom, t, g)["wald_equal_pvalue"] > 0.001

    def test_sensitivity_rv_bounds(self):
        from social_trading_dml.diagnostics.sensitivity import robustness_value, sensitivity_analysis
        assert robustness_value(0) == 0.0
        assert 0 < robustness_value(0.1) < 1
        rng = np.random.default_rng(0)
        n = 2000
        W = rng.normal(size=(n, 2))
        T = 0.5 * W[:, 0] + rng.normal(size=n)       # w0 explains ~20% of T
        Y = -1.0 * T + 0.5 * W[:, 0] + rng.normal(size=n)
        from social_trading_dml.estimation.dml import clustered_final_stage
        dml_like = clustered_final_stage(Y - 0.5 * W[:, 0], T - 0.5 * W[:, 0])
        out = sensitivity_analysis(
            {"Y": Y, "T": T, "W": W, "W_names": ["w0", "w1"]}, dml_like)
        assert 0 < out["rv_q1_alpha"] <= out["rv_q1"] < 1
        bench = {b["control"]: b for b in out["benchmarks"]}
        # w1 is pure noise: a confounder as strong as w1 implies (almost) no bias,
        # far less than one as strong as the real confounder w0
        assert bench["w1"]["bias"] < 0.1 * bench["w0"]["bias"]
        assert not bench["w1"]["sign_flips"]


class TestSubsampling:
    def test_subsample_by_trader_preserves_lags(self):
        """Sampling whole traders keeps _l1 equal to the previous week's value."""
        from social_trading_dml.data.features import compute_lags
        from tests.synthetic import make_test_panel
        df = make_test_panel(n_traders=20, n_periods=8)
        keep = np.random.RandomState(0).choice(df.trader_id.unique(), 5, replace=False)
        sub = compute_lags(df[df.trader_id.isin(keep)], lag_columns=["learner_count"])
        for _, g in sub.groupby("trader_id"):
            g = g.sort_values("period")
            np.testing.assert_array_equal(g["learner_count_l1"].values[1:],
                                          g["learner_count"].values[:-1])


class TestRegressions:
    """Regression tests for issues found in code review (v2 -> v3)."""

    def test_rv_matches_cinelli_hazlett_darfur(self):
        """Cinelli & Hazlett (2020) Darfur example: t=4.18, df=783 gives
        RV_q=1 = 13.9% and RV_q=1,alpha=0.05 = 7.6%."""
        from social_trading_dml.diagnostics.sensitivity import sensitivity_analysis
        rng = np.random.default_rng(0)
        n = 500
        W = rng.normal(size=(n, 1))
        T = rng.normal(size=n)
        Y = T + rng.normal(size=n)
        est = 0.097
        dml_like = {"t_stat_classical": 4.18, "dof": 783, "ate": est,
                    "ate_se": est / 4.18, "partial_r2_ty": 4.18 ** 2 / (4.18 ** 2 + 783)}
        out = sensitivity_analysis({"Y": Y, "T": T, "W": W, "W_names": ["w"]}, dml_like)
        assert out["rv_q1"] == pytest.approx(0.139, abs=0.001)
        assert out["rv_q1_alpha_classical_se"] == pytest.approx(0.076, abs=0.001)
        # Clustered SE equal to the classical SE gives the same RV_alpha
        assert out["rv_q1_alpha"] == pytest.approx(out["rv_q1_alpha_classical_se"])

    def test_rv_alpha_shrinks_with_clustered_se(self):
        from social_trading_dml.diagnostics.sensitivity import sensitivity_analysis
        rng = np.random.default_rng(0)
        W = rng.normal(size=(300, 1))
        T = rng.normal(size=300)
        Y = T + rng.normal(size=300)
        arrays = {"Y": Y, "T": T, "W": W, "W_names": ["w"]}
        base = {"t_stat_classical": 10.0, "dof": 10000, "ate": 1.0, "partial_r2_ty": 0.01}
        a = sensitivity_analysis(arrays, {**base, "ate_se": 0.1})
        b = sensitivity_analysis(arrays, {**base, "ate_se": 0.3})
        assert a["rv_q1"] == b["rv_q1"]
        assert b["rv_q1_alpha"] < a["rv_q1_alpha"]

    def test_trader_folds_never_split_a_trader(self):
        from social_trading_dml.estimation.dml import trader_folds
        groups = np.repeat(np.arange(103), np.random.default_rng(0).integers(1, 9, 103))
        folds = trader_folds(groups, n_folds=5, random_state=1)
        test_idx = np.concatenate([te for _, te in folds])
        assert np.array_equal(np.sort(test_idx), np.arange(len(groups)))
        for train, test in folds:
            assert not set(groups[train]) & set(groups[test])
            assert len(np.intersect1d(train, test)) == 0

    def test_fit_linear_dml_uses_trader_folds(self, monkeypatch):
        """fit_linear_dml must hand trader_folds() splits to econml."""
        import social_trading_dml.estimation.dml as m
        from social_trading_dml.estimation.panel import build_fe_arrays
        seen = {}
        real = m.trader_folds

        def spy(groups, n_folds, random_state):
            seen["folds"] = real(groups, n_folds, random_state)
            return seen["folds"]

        monkeypatch.setattr(m, "trader_folds", spy)
        a = build_fe_arrays(simulate_panel(60, 6), "T", "Y", ["x"])
        m.fit_linear_dml(a["Y"], a["T"], a["W"], groups=a["groups"], nuisance_learner="lasso")
        assert "folds" in seen
        for train, test in seen["folds"]:
            assert not set(a["groups"][train]) & set(a["groups"][test])

    def test_absorbed_fractional_tenure_dropped_unbalanced(self):
        """Tenure = week - join date (fractional) is collinear with trader + week
        FE; it must be dropped even when the panel is unbalanced."""
        from social_trading_dml.estimation.panel import build_fe_arrays
        df = simulate_panel(80, 12).sample(frac=0.7, random_state=3)
        join = (df["trader_id"] * 0.37) % 5
        df["tenure_weeks"] = df["period"] - 20 + join
        a = build_fe_arrays(df, "T", "Y", ["x", "tenure_weeks"])
        assert a["W_names"] == ["x"]

    def test_trim_extreme_treatment_only_trims_treatment(self):
        from social_trading_dml.diagnostics.robustness import trim_extreme_treatment
        rng = np.random.default_rng(0)
        T = np.where(rng.random(20000) < 0.1, rng.integers(1, 50, 20000), 0)
        df = pd.DataFrame({"total_self_trxn": T, "total_gain_w": rng.normal(0, 1e6, 20000)})
        out, stats = trim_extreme_treatment(df, quantile=0.99)
        pos = df.loc[df.total_self_trxn > 0, "total_self_trxn"]
        assert stats["cutoff"] == pytest.approx(pos.quantile(0.99))
        assert (out["total_self_trxn"] <= stats["cutoff"]).all()
        assert stats["n_removed"] <= 0.01 * len(pos) + 1  # outcome never used

    def test_repair_audience_gaps(self):
        from social_trading_dml.data.features import repair_audience_gaps
        rows = [{"trader_id": tid, "period": p,
                 "learner_count": 0 if p == 33 else tid + p}
                for tid in range(1, 21) for p in range(30, 36)]
        out, gaps = repair_audience_gaps(pd.DataFrame(rows), cols=["learner_count"])
        assert gaps["learner_count"] == [33]
        week33 = out[out.period == 33].set_index("trader_id")["learner_count"]
        week32 = out[out.period == 32].set_index("trader_id")["learner_count"]
        assert (week33 == week32).all()       # carried forward, not zero
        assert out.loc[out.period == 33, "audience_imputed"].eq(1).all()
        assert out.loc[out.period != 33, "audience_imputed"].eq(0).all()

    def test_mid_week_joiner_is_not_pre_join(self):
        from social_trading_dml.data.features import compute_tenure
        df = pd.DataFrame({
            "trader_id": [1, 1, 1],
            "period": [30, 31, 32],
            "min_period_date": pd.to_datetime(["2013-01-01", "2013-01-08", "2013-01-15"]),
            "max_period_date": pd.to_datetime(["2013-01-07", "2013-01-14", "2013-01-21"]),
            "joining_date": pd.to_datetime(["2013-01-10"] * 3),
        })
        out = compute_tenure(df)
        assert out["pre_join"].tolist() == [True, False, False]
        assert (out["tenure_weeks"] >= 0).all()

    def test_winsorize_components_on_own_nonzero_values(self):
        """Self gain is zero in most active weeks; its bounds must come from its
        own non-zero values, not from all active weeks."""
        from social_trading_dml.data.variables import compute_outcome
        rng = np.random.default_rng(0)
        n = 20000
        self_gain = np.where(rng.random(n) < 0.25, rng.normal(0, 100, n), 0.0)
        copied = rng.normal(0, 50, n)
        df = pd.DataFrame({"total_trxn": np.ones(n), "total_self_gain": self_gain,
                           "total_copied_gain": copied, "total_gain": self_gain + copied})
        out = compute_outcome(df)
        nz = self_gain[self_gain != 0]
        assert out["total_self_gain_w"].max() == pytest.approx(np.quantile(nz, 0.99), rel=1e-6)
        clipped = (out["total_self_gain_w"] != df["total_self_gain"])[self_gain != 0].mean()
        assert clipped < 0.03

    def test_week_decomposition_identity(self):
        from social_trading_dml.diagnostics.identification import week_decomposition
        rng = np.random.default_rng(0)
        t = rng.normal(size=1000)
        y = -2 * t + rng.normal(size=1000)
        periods = rng.integers(30, 40, 1000)
        d = week_decomposition(y, t, periods)
        theta = (t @ y) / (t @ t)
        assert d["weight"].sum() == pytest.approx(1.0)
        assert (d["weight"] * d["slope"]).sum() == pytest.approx(theta)
        m = periods != d["period"].iloc[0]
        assert d["estimate_leave_week_out"].iloc[0] == pytest.approx(
            (t[m] @ y[m]) / (t[m] @ t[m]))

    def test_week_decomposition_se_matches_hc0(self):
        """Per-week SE equals the HC0 robust SE of a no-intercept OLS within the week."""
        import statsmodels.api as sm
        from social_trading_dml.diagnostics.identification import week_decomposition
        rng = np.random.default_rng(1)
        t = rng.normal(size=600)
        y = -2 * t + rng.normal(size=600) * (1 + np.abs(t))
        periods = np.repeat([30, 31, 32], 200)
        d = week_decomposition(y, t, periods).set_index("period")
        w = periods == 31
        fit = sm.OLS(y[w], t[w]).fit(cov_type="HC0")
        assert d.loc[31, "slope"] == pytest.approx(fit.params[0])
        assert d.loc[31, "slope_se"] == pytest.approx(fit.bse[0])

    def test_group_effects_recovers_group_slopes(self):
        from social_trading_dml.estimation.heterogeneity import group_effects
        rng = np.random.default_rng(0)
        n = 6000
        lab = np.where(rng.random(n) < 0.5, "a", "b")
        t = rng.normal(size=n)
        y = np.where(lab == "a", -1.0, 3.0) * t + rng.normal(size=n)
        res = group_effects(lab, y, t, np.arange(n) // 5)
        eff = {g["group"]: g["effect"] for g in res["groups"]}
        assert eff["a"] == pytest.approx(-1.0, abs=0.1)
        assert eff["b"] == pytest.approx(3.0, abs=0.1)
        assert res["wald_equal_pvalue"] < 1e-10

    def test_two_way_clustered_se_reported(self):
        from social_trading_dml.estimation.dml import clustered_final_stage
        rng = np.random.default_rng(0)
        g = np.repeat(np.arange(200), 10)
        w = np.tile(np.arange(10), 200)
        shock = rng.normal(size=10)[w]          # common week shock in residuals
        t = rng.normal(size=2000) + shock
        y = -1 * t + 3 * shock + rng.normal(size=2000)
        res = clustered_final_stage(y, t, g, w)
        assert res["n_time_clusters"] == 10
        assert res["ate_se_twoway"] > res["ate_se"]

    def test_placebo_within_trader_is_null(self):
        from social_trading_dml.diagnostics.robustness import placebo_within_trader
        df = simulate_panel(200, 10, theta=-2.0).rename(
            columns={"T": "total_self_trxn", "Y": "total_gain_w"})
        r = placebo_within_trader(df, ["x"], n_reps=1).iloc[0]
        assert abs(r["ate"]) < 0.3
