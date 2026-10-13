"""
Synthetic panels for the test suite.

- make_test_panel(): small trader-week panel with the raw column layout of
  the XTrader data (for feature/sample-construction tests)
- simulate_panel(): panel with a known treatment effect, trader effects and
  week shocks that confound treatment and outcome (for estimator tests)
"""

import numpy as np
import pandas as pd


def make_test_panel(n_traders=5, n_periods=10, seed=0):
    """Create a small test panel for unit tests."""
    np.random.seed(seed)
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


def simulate_panel(n_traders=300, n_periods=20, theta=-2.0, seed=0):
    rng = np.random.default_rng(seed)
    tid = np.repeat(np.arange(n_traders), n_periods)
    period = np.tile(np.arange(30, 30 + n_periods), n_traders)
    alpha = rng.normal(0, 3, n_traders)[tid]            # trader effect (skill)
    gamma = rng.normal(0, 3, n_periods)[period - 30]    # week shock (market)
    x = rng.normal(size=len(tid))                        # observed confounder
    T = 1.0 * alpha + 1.0 * gamma + x + rng.normal(size=len(tid))
    Y = theta * T + 2.0 * alpha + 2.0 * gamma + np.sin(x) * 2 + rng.normal(size=len(tid))
    return pd.DataFrame({"trader_id": tid, "period": period, "T": T, "Y": Y, "x": x})
