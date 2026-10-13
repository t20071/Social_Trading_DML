"""
Feedback-robust estimation.

The lead test rejects strict exogeneity: gains in one week predict
self-trading in later weeks. Within (fixed-effects) and first-difference
estimators then are biased. Under the weaker *sequential* exogeneity
assumption -- this week's shock is unrelated to self-trading up to and
including this week, but may affect later self-trading -- the
Anderson-Hsiao estimator is consistent:

    dY_it = theta * dT_it + d_lambda_t + d_eps_it          (first differences)

with dT_it = T_it - T_i,t-1 instrumented by T_i,t-2, which is unrelated to
eps_it and eps_i,t-1 under sequential exogeneity (Anderson & Hsiao 1982;
Arellano & Bond 1991 use all such lags in GMM). Week effects are removed by
demeaning the differenced data by week.

Caveats: (i) the exclusion restriction fails if trades opened two weeks
earlier book their P&L in t or t-1; (ii) no time-varying controls are
included, so this is a linear benchmark, not a replacement for the DML
specification.
"""

import logging
from typing import Any, Dict

import numpy as np
import pandas as pd

from social_trading_dml.config import MIN_CLUSTERS_FOR_CRVE

logger = logging.getLogger(__name__)


def _demean_by(values: np.ndarray, keys: np.ndarray) -> np.ndarray:
    s = pd.Series(values)
    return (s - s.groupby(keys).transform("mean")).to_numpy()


def _crve_se(z: np.ndarray, e: np.ndarray, zx: float, clusters: np.ndarray) -> float:
    """Cluster-robust SE of a just-identified IV coefficient (CRV1)."""
    codes = pd.factorize(clusters)[0]
    g = int(codes.max()) + 1
    score = np.bincount(codes, weights=z * e, minlength=g)
    return float(np.sqrt(g / (g - 1) * np.sum(score ** 2)) / abs(zx))


def anderson_hsiao(
    df: pd.DataFrame,
    outcome: str,
    outcome_lag: str,
    treatment: str,
    treatment_lag: str,
    instrument: str,
    group_col: str = "trader_id",
    time_col: str = "period",
) -> Dict[str, Any]:
    """
    Just-identified IV in first differences with week effects.

    All lags (`outcome_lag`, `treatment_lag`, `instrument`) should be computed
    on the full panel before sample filters, so no rows are lost to
    differencing. Returns theta with trader-, week- and two-way-clustered SEs
    and the first-stage F statistic (trader-clustered).
    """
    cols = list(dict.fromkeys([group_col, time_col, outcome, outcome_lag,
                               treatment, treatment_lag, instrument]))
    d = df[cols].dropna()
    weeks = d[time_col].to_numpy()
    dy = _demean_by((d[outcome] - d[outcome_lag]).to_numpy(np.float64), weeks)
    dx = _demean_by((d[treatment] - d[treatment_lag]).to_numpy(np.float64), weeks)
    z = _demean_by(d[instrument].to_numpy(np.float64), weeks)

    zx = float(z @ dx)
    theta = float(z @ dy) / zx
    e = dy - theta * dx

    traders = d[group_col].to_numpy()
    se_trader = _crve_se(z, e, zx, traders)
    se_week = _crve_se(z, e, zx, weeks)
    # two-way: trader + week - intersection (trader-week = one row -> HC0)
    v_cell = float(np.sum((z * e) ** 2)) / zx ** 2
    se_twoway = float(np.sqrt(max(se_trader ** 2 + se_week ** 2 - v_cell, 0.0)))

    # first stage: dx on z, trader-clustered
    pi = float(z @ dx) / float(z @ z)
    u = dx - pi * z
    se_pi = _crve_se(z, u, float(z @ z), traders)
    n_weeks = int(len(np.unique(weeks)))

    out = {
        "theta": theta,
        "se_trader": se_trader,
        "ci_lower_trader": theta - 1.96 * se_trader,
        "ci_upper_trader": theta + 1.96 * se_trader,
        "se_week": se_week if n_weeks >= MIN_CLUSTERS_FOR_CRVE else None,
        "se_twoway": se_twoway if n_weeks >= MIN_CLUSTERS_FOR_CRVE else None,
        "first_stage_coef": pi,
        "first_stage_F": float((pi / se_pi) ** 2),
        "n": int(len(d)),
        "n_traders": int(len(np.unique(traders))),
        "n_weeks": n_weeks,
        "instrument": instrument,
    }
    if out["se_twoway"] is not None:
        out["ci_lower_twoway"] = theta - 1.96 * se_twoway
        out["ci_upper_twoway"] = theta + 1.96 * se_twoway
    logger.info(
        f"Anderson-Hsiao ({instrument}): theta={theta:.3f} "
        f"(trader SE {se_trader:.3f}, two-way SE {se_twoway:.3f}), "
        f"first-stage F={out['first_stage_F']:.0f}, N={out['n']:,}"
    )
    return out
