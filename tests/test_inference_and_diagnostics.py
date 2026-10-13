"""
Tests for week-level inference, the feedback-robust estimator, the
heterogeneity tests and the identification diagnostics.

Each test checks a property with a known answer on simulated data.
"""

import json

import numpy as np
import pandas as pd
import pytest

from tests.synthetic import simulate_panel


def feedback_panel(n_traders=1500, n_periods=12, theta=-2.0, rho=0.8, phi=0.6, seed=0):
    """
    Panel with feedback: this week's gain shock raises NEXT week's
    self-trading (T_t depends on eps_{t-1}), so strict exogeneity fails but
    sequential exogeneity holds. Self-trading is persistent (T_t depends on
    T_{t-1} with coefficient phi); without persistence a lagged level carries
    no information about the change dT and Anderson-Hsiao is not identified.
    """
    rng = np.random.default_rng(seed)
    alpha = rng.normal(0, 1, n_traders)
    lam = rng.normal(0, 1, n_periods)
    rows = []
    for i in range(n_traders):
        eps_prev, T = 0.0, 2.0 + alpha[i]
        for t in range(n_periods):
            eps = rng.normal()
            T = (1 - phi) * (2.0 + alpha[i]) + phi * T + rho * eps_prev + rng.normal()
            Y = theta * T + 2.0 * alpha[i] + lam[t] + eps
            rows.append((i, 30 + t, T, Y))
            eps_prev = eps
    df = pd.DataFrame(rows, columns=["trader_id", "period", "total_self_trxn", "total_gain_w"])
    df = df.sort_values(["trader_id", "period"])
    g = df.groupby("trader_id")
    df["total_self_trxn_l1"] = g["total_self_trxn"].shift(1)
    df["total_self_trxn_l2"] = g["total_self_trxn"].shift(2)
    df["total_gain_w_l1"] = g["total_gain_w"].shift(1)
    df["total_self_trxn_f1"] = g["total_self_trxn"].shift(-1)
    return df


class TestWildClusterBootstrap:
    def test_estimate_matches_ols_and_effective_clusters(self):
        from social_trading_dml.estimation.inference import (
            effective_clusters, wild_cluster_bootstrap)
        rng = np.random.default_rng(0)
        x = rng.normal(size=2000)
        y = 1.5 * x + rng.normal(size=2000)
        c = np.repeat(np.arange(20), 100)
        out = wild_cluster_bootstrap(y, x, c, reps=999)
        assert out["theta"] == pytest.approx((x @ y) / (x @ x))
        assert out["n_clusters"] == 20
        assert effective_clusters(x, c) == pytest.approx(20, rel=0.2)
        assert out["p_wild"] < 0.01
        assert out["ci_lower_wild"] < out["theta"] < out["ci_upper_wild"]

    def test_ci_inverts_the_wcr_test(self):
        """The CI is the set of theta0 the WCR test does not reject, so it
        excludes 0 exactly when p < 0.05, and p crosses 0.05 at its ends."""
        import social_trading_dml.estimation.inference as inf
        for seed, beta in [(0, 0.0), (1, 0.15), (2, 0.4)]:
            rng = np.random.default_rng(seed)
            c = np.repeat(np.arange(12), 150)
            x = rng.normal(size=12)[c] + rng.normal(size=1800)
            y = beta * x + rng.normal(size=12)[c] + rng.normal(size=1800)
            out = inf.wild_cluster_bootstrap(y, x, c, reps=999, seed=seed)
            lo, hi = out["ci_lower_wild"], out["ci_upper_wild"]
            assert lo < out["theta"] < hi
            assert (lo > 0 or hi < 0) == (out["p_wild"] < 0.05)
            shifted = lambda t0: inf.wild_cluster_bootstrap(y - t0 * x, x, c, reps=999, seed=seed)["p_wild"]
            width = hi - lo
            assert shifted(lo + 0.01 * width) >= 0.05 > shifted(lo - 0.01 * width)
            assert shifted(hi - 0.01 * width) >= 0.05 > shifted(hi + 0.01 * width)

    def test_effective_clusters_detects_concentration(self):
        from social_trading_dml.estimation.inference import effective_clusters
        x = np.r_[np.full(100, 10.0), np.full(1900, 0.1)]
        c = np.r_[np.zeros(100), np.repeat(np.arange(1, 20), 100)]
        assert effective_clusters(x, c) < 1.5

    def test_null_is_not_rejected_too_often(self):
        """Under H0 with common cluster shocks, the week bootstrap keeps size;
        trader-style i.i.d. inference would reject far too often."""
        from social_trading_dml.estimation.inference import wild_cluster_bootstrap
        rejections = 0
        for seed in range(40):
            rng = np.random.default_rng(seed)
            c = np.repeat(np.arange(15), 200)
            shock_x, shock_y = rng.normal(size=15)[c], rng.normal(size=15)[c]
            x = shock_x + 0.3 * rng.normal(size=3000)
            y = shock_y + 0.3 * rng.normal(size=3000)
            rejections += wild_cluster_bootstrap(y, x, c, reps=499, seed=seed)["p_wild"] < 0.05
        assert rejections <= 6        # nominal 5% of 40 = 2

    def test_week_inference_not_reported_with_few_clusters(self):
        from social_trading_dml.estimation.dml import clustered_final_stage
        rng = np.random.default_rng(0)
        t = rng.normal(size=600)
        y = t + rng.normal(size=600)
        res = clustered_final_stage(y, t, np.arange(600) // 6, np.repeat([1, 2, 3], 200))
        assert res["n_time_clusters"] == 3
        assert res["ate_ci_lower_week_wild"] is None and res["ate_se_twoway"] is None

    def test_linear_fe_ols_matches_full_regression(self):
        import statsmodels.api as sm
        from social_trading_dml.estimation.dml import linear_fe_ols
        from social_trading_dml.estimation.panel import build_fe_arrays
        df = simulate_panel(100, 12)
        a = build_fe_arrays(df, "T", "Y", ["x"])
        res = linear_fe_ols(a)
        full = sm.OLS(a["Y"], np.column_stack([a["T"], a["W"]])).fit()
        assert res["ate"] == pytest.approx(full.params[0])
        assert res["ate_ci_lower_week_wild"] is not None


class TestFeedback:
    def test_lead_test_detects_feedback(self):
        from social_trading_dml.diagnostics.identification import lead_test
        df = feedback_panel(n_traders=800, n_periods=10)
        df["x"] = np.random.default_rng(1).normal(size=len(df))
        out = lead_test(df, ["x"])
        assert out["linear_fe_lead_coef"] > 0
        assert out["linear_fe_lead_pvalue"] < 0.001

    def test_anderson_hsiao_recovers_effect_under_feedback(self):
        from social_trading_dml.estimation.dynamic import anderson_hsiao
        df = feedback_panel()
        ah = anderson_hsiao(df, "total_gain_w", "total_gain_w_l1", "total_self_trxn",
                            "total_self_trxn_l1", "total_self_trxn_l2")
        assert ah["theta"] == pytest.approx(-2.0, abs=4 * ah["se_trader"])
        assert abs(ah["theta"] + 2.0) < 0.3
        assert ah["first_stage_F"] > 10


class TestHeterogeneityTests:
    def test_blp_calibration(self):
        from social_trading_dml.estimation.heterogeneity import blp_calibration
        rng = np.random.default_rng(0)
        n = 20000
        tau = rng.normal(-2, 1.5, n)
        t = rng.normal(size=n)
        y = tau * t + rng.normal(size=n)
        g = np.arange(n) // 10
        good = blp_calibration(tau, y, t, g)
        noise = blp_calibration(rng.normal(size=n), y, t, g)
        assert good["b2_het"] == pytest.approx(1.0, abs=0.1)
        assert abs(noise["b2_het"]) < 0.1 and noise["b2_pvalue"] > 0.01

    def test_blp_moderators_net_of_block_intercepts(self):
        """A moderator that only proxies for calendar blocks must show no effect
        once block intercepts are included; a real moderator must."""
        from social_trading_dml.estimation.heterogeneity import blp_moderators
        rng = np.random.default_rng(0)
        n = 30000
        block = np.where(rng.random(n) < 0.5, "early", "late")
        real = rng.normal(size=n)
        proxy = (block == "late") + 0.1 * rng.normal(size=n)
        tau = np.where(block == "early", -5.0, 0.0) + 1.0 * real
        t = rng.normal(size=n)
        y = tau * t + rng.normal(size=n)
        out = blp_moderators(pd.DataFrame({"real": real, "proxy": proxy}), block, y, t,
                             np.arange(n) // 10)
        coef = {c["term"]: c for c in out["terms"]}
        assert coef["real"]["pvalue"] < 1e-6
        assert coef["proxy"]["pvalue"] > 0.01


class TestDiagnostics:
    def test_overlap_check_keeps_binary_controls(self):
        from social_trading_dml.diagnostics.identification import overlap_check
        rng = np.random.default_rng(0)
        W = np.column_stack([rng.normal(size=500), (rng.random(500) < 0.1).astype(float)])
        out = overlap_check(rng.poisson(1, 500), W, feature_names=["cont", "binary"])
        assert set(out["confounder"]) == {"cont", "binary"}
        assert set(out.loc[out.confounder == "binary", "bin"]) == {"zero", "value 1"}

    def test_regime_break_and_aggregates(self):
        from social_trading_dml.diagnostics.identification import regime_break, week_aggregates
        periods = np.arange(31, 61)
        slope = np.where(periods <= 45, -20.0, -1.0)
        wd = pd.DataFrame({"period": periods, "slope": slope, "slope_se": 1.0,
                           "weight": np.where(periods <= 45, 0.01, 0.05)})
        wd["weight"] /= wd["weight"].sum()
        br = regime_break(wd)
        assert br["break_after_period"] == 45
        assert br["mean_before"] == pytest.approx(-20) and br["mean_after"] == pytest.approx(-1)
        agg = week_aggregates(wd, pd.Series(1.0, index=periods))
        assert agg["equal_week"] == pytest.approx(-10.5)
        assert agg["variance_weighted"] == pytest.approx((wd.weight * wd.slope).sum())


class TestDataHelpers:
    def test_partial_gap_week_keeps_recorded_values(self):
        from social_trading_dml.data.features import repair_audience_gaps
        rows = []
        for tid in range(1, 21):
            for p in range(30, 36):
                v = tid + p
                if p == 33 and tid > 3:      # week 33: only traders 1-3 recorded
                    v = 0
                rows.append({"trader_id": tid, "period": p, "learner_count": v})
        out, gaps = repair_audience_gaps(pd.DataFrame(rows), cols=["learner_count"])
        assert gaps["learner_count"] == [33]
        w33 = out[out.period == 33].set_index("trader_id")
        assert w33.loc[1, "learner_count"] == 1 + 33          # recorded value kept
        assert w33.loc[1, "audience_imputed"] == 0
        assert w33.loc[10, "learner_count"] == 10 + 32        # zero filled from week 32
        assert w33.loc[10, "audience_imputed"] == 1

    def test_pre_window_moderators(self):
        from social_trading_dml.data.features import add_pre_window_moderators
        df = pd.DataFrame({
            "trader_id": [1, 1, 2, 2], "period": [30, 31, 30, 31],
            "min_period_date": pd.to_datetime(["2013-01-01", "2013-01-08"] * 2),
            "joining_date": pd.to_datetime(["2012-12-18"] * 2 + ["2013-01-15"] * 2),
            "learner_count": [5, 9, 0, 1], "total_follower_count": [7, 7, 2, 3],
        })
        out = add_pre_window_moderators(df).set_index(["trader_id", "period"])
        assert out.loc[(1, 31), "join_cohort_weeks"] == pytest.approx(2.0)
        assert out.loc[(2, 30), "join_cohort_weeks"] == pytest.approx(-2.0)
        assert out.loc[(1, 31), "copiers_start"] == 5    # period-30 value, not current
        assert out.loc[(2, 31), "followers_start"] == 2

    def test_to_jsonable_and_manifest(self, tmp_path):
        from social_trading_dml.io import save_json, to_jsonable, update_manifest
        obj = {"a": np.float64(1.5), "b": np.int64(2), "big": np.zeros(100), "small": np.ones(3)}
        out = to_jsonable(obj)
        assert out == {"a": 1.5, "b": 2, "small": [1.0, 1.0, 1.0]}
        save_json(obj, "x.json", tmp_path)
        update_manifest(tmp_path, ["x.json"], {"run_started": "r1"})
        update_manifest(tmp_path, ["y.json"], {"run_started": "r2"})
        m = json.loads((tmp_path / "manifest.json").read_text())
        assert m["files"] == {"x.json": "r1", "y.json": "r2"} and m["latest_run"] == "r2"


class TestWaldPSD:
    def test_matches_chi2_when_psd_and_survives_negative_eigenvalue(self):
        import statsmodels.api as sm
        from scipy import stats
        from social_trading_dml.estimation.heterogeneity import wald_psd
        rng = np.random.default_rng(0)
        X = rng.normal(size=(500, 3))
        y = X @ np.array([0.3, 0.0, -0.2]) + rng.normal(size=500)
        fit = sm.OLS(y, X).fit()
        R = np.eye(3)[[0, 2]]
        b, V = R @ fit.params, R @ fit.cov_params() @ R.T
        assert wald_psd(fit, R)["pvalue"] == pytest.approx(stats.chi2.sf(b @ np.linalg.solve(V, b), 2))

        class Fake:  # covariance with a negative eigenvalue, as two-way CRVE can produce
            params = np.array([1.0, 1.0])
            def cov_params(self):
                return np.array([[1.0, 1.2], [1.2, 1.0]])   # eigenvalues 2.2 and -0.2
        out = wald_psd(Fake(), np.eye(2))
        assert out["n_negative_eigenvalues"] == 1 and out["df"] == 1
        assert 0 < out["pvalue"] < 1 and out["stat"] > 0
