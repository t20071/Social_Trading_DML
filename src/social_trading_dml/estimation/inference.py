"""
Inference for a single partialled-out coefficient when the identifying
variation is concentrated in a few clusters (here: a few calendar weeks).

- wild_cluster_bootstrap(): wild cluster bootstrap for y = theta * x + e
  (the DML final stage, or any coefficient after Frisch-Waugh-Lovell
  partialling). p-value from the restricted bootstrap (WCR, H0: theta = 0);
  CI by inverting the same WCR test, so the CI excludes 0 exactly when
  p < alpha. Webb six-point weights, which behave better than Rademacher
  with few clusters (Cameron, Gelbach & Miller 2008; MacKinnon, Nielsen &
  Webb 2023).
- effective_clusters(): Kish effective number of clusters, 1 / sum(w_g^2),
  with w_g the cluster's share of sum(x^2). A heuristic for how many
  clusters actually identify theta.

All quantities are computed from per-cluster sums, so B = 9,999 draws cost
milliseconds even with a million observations.
"""

from typing import Any, Dict, Optional

import numpy as np
import pandas as pd

WEBB_WEIGHTS = np.array([-np.sqrt(1.5), -1.0, -np.sqrt(0.5), np.sqrt(0.5), 1.0, np.sqrt(1.5)])


def _cluster_sums(x: np.ndarray, y: np.ndarray, clusters: np.ndarray):
    codes = pd.factorize(np.asarray(clusters))[0]
    n_clusters = int(codes.max()) + 1
    sxx = np.bincount(codes, weights=x * x, minlength=n_clusters)
    sxy = np.bincount(codes, weights=x * y, minlength=n_clusters)
    return sxx, sxy, n_clusters


def effective_clusters(x: np.ndarray, clusters: np.ndarray) -> float:
    """Kish effective number of clusters for the coefficient on x."""
    x = np.asarray(x, dtype=np.float64).ravel()
    sxx, _, _ = _cluster_sums(x, np.zeros_like(x), clusters)
    w = sxx / sxx.sum()
    return float(1.0 / np.sum(w ** 2))


def wild_cluster_bootstrap(
    y: np.ndarray,
    x: np.ndarray,
    clusters: np.ndarray,
    reps: int = 9999,
    alpha: float = 0.05,
    seed: int = 42,
) -> Dict[str, Any]:
    """
    Wild cluster bootstrap for theta in y = theta * x + e (no intercept).

    Returns the estimate, the one-way cluster-robust (CRV1) SE and t-statistic,
    the WCR bootstrap p-value for H0: theta = 0, the confidence interval from
    inverting the WCR test (the set of theta0 with p(theta0) >= alpha, using
    the same draws for every theta0), the number of clusters and the
    effective number of clusters.
    """
    y = np.asarray(y, dtype=np.float64).ravel()
    x = np.asarray(x, dtype=np.float64).ravel()
    sxx, sxy, n_clusters = _cluster_sums(x, y, clusters)
    sxx_total = sxx.sum()
    theta = sxy.sum() / sxx_total

    n = len(y)
    small_sample = n_clusters / (n_clusters - 1)               # CRV1 with one regressor
    score = sxy - theta * sxx                                   # sum_i x_i e_i by cluster
    se = np.sqrt(small_sample * np.sum(score ** 2)) / sxx_total
    t_stat = theta / se

    rng = np.random.default_rng(seed)
    draws = rng.choice(WEBB_WEIGHTS, size=(reps, n_clusters))

    def p_wcr(theta0: float) -> float:
        # Restricted residual y - theta0 * x has cluster score sums sxy - theta0 * sxx
        s0 = sxy - theta0 * sxx
        th_r = draws @ s0 / sxx_total
        score_r = draws * s0 - th_r[:, None] * sxx
        se_r = np.sqrt(small_sample * np.sum(score_r ** 2, axis=1)) / sxx_total
        return float(np.mean(np.abs(th_r / se_r) >= abs(theta - theta0) / se))

    def ci_bound(direction: int) -> float:
        # Expand until the test rejects, then bisect; p(theta0) is a step function
        inner, step = theta, se
        outer = theta + direction * step
        while p_wcr(outer) >= alpha:
            inner, step = outer, 2 * step
            outer = theta + direction * step
            if step > 1e6 * se:
                return float(direction * np.inf)
        for _ in range(60):
            mid = 0.5 * (inner + outer)
            if p_wcr(mid) >= alpha:
                inner = mid
            else:
                outer = mid
        return float(inner)

    w = sxx / sxx_total
    return {
        "theta": float(theta),
        "se_cluster": float(se),
        "t_cluster": float(t_stat),
        "p_wild": p_wcr(0.0),
        "ci_lower_wild": ci_bound(-1),
        "ci_upper_wild": ci_bound(+1),
        "n_clusters": n_clusters,
        "effective_clusters": float(1.0 / np.sum(w ** 2)),
        "reps": reps,
        "weights": "webb",
    }


def partial_out(target: np.ndarray, controls: Optional[np.ndarray]) -> np.ndarray:
    """OLS residual of `target` on `controls` (Frisch-Waugh-Lovell), no intercept."""
    target = np.asarray(target, dtype=np.float64)
    if controls is None or controls.shape[1] == 0:
        return target
    coef, *_ = np.linalg.lstsq(controls, target, rcond=None)
    return target - controls @ coef
