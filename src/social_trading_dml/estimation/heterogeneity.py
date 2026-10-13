
"""
Treatment-effect heterogeneity.

- fit_causal_forest(): CausalForestDML on the two-way FE data
- group_effects(): held-out average effect within groups of any label, with
  trader-clustered SEs and a Wald test of equal effects
- gates_test(): GATES by quantile of predicted CATE (optionally within
  period blocks)
- blp_calibration(): best-linear-predictor test of whether the forest
  predictions carry real heterogeneity (held-out data)
- blp_moderators(): trader-level moderators with period-block intercepts
"""

import logging
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from social_trading_dml.config import MIN_CLUSTERS_FOR_CRVE
from social_trading_dml.estimation.cache import fingerprint, load_cache, save_cache
from social_trading_dml.estimation.dml import build_nuisance_learners, trader_folds

logger = logging.getLogger(__name__)


def fit_causal_forest(
    Y: np.ndarray,
    T: np.ndarray,
    X: np.ndarray,
    W: np.ndarray,
    n_estimators: int = 200,
    min_samples_leaf: int = 50,
    cv: int = 5,
    random_state: int = 42,
    cache_name: Optional[str] = None,
    force_refit: bool = False,
    max_samples: Optional[int] = None,
    groups: Optional[np.ndarray] = None,
) -> Tuple[Any, Dict[str, Any], np.ndarray]:
    """
    Fit CausalForestDML for treatment-effect heterogeneity (CATE).

    With `groups`, subsampling for fitting is done by whole traders and the
    nuisance cross-fitting uses trader_folds(). Note: econml's forest stage
    and its BLB intervals (effect_interval) still treat rows as independent.
    `feature_importances_` is split-based and descriptive, not a test of
    heterogeneity — use gates_test() / group_effects() on held-out data.

    Parameters
    ----------
    Y : array-like, shape (n,)
        Outcome.
    T : array-like, shape (n,)
        Treatment.
    X : array-like, shape (n, q)
        Heterogeneity features (learner_count_l1, total_follower_count_l1,
        tenure_weeks).
    W : array-like, shape (n, p)
        Nuisance-only confounders.
    n_estimators : int
        Number of trees in the causal forest.
    min_samples_leaf : int
        Minimum samples per leaf.
    cv : int
        Cross-fitting folds.
    random_state : int
        Random seed.
    cache_name : str, optional
        Cache name.
    force_refit : bool
        Force refit.
    max_samples : int, optional
        If provided, subsample for faster fitting.

    Returns
    -------
    Tuple[CausalForestDML, Dict, np.ndarray]
        Fitted model, summary results, and CATE predictions.
    """
    from econml.dml import CausalForestDML

    model_y, model_t = build_nuisance_learners("histgbr", random_state)
    fp = fingerprint(Y, T, X, W, groups, n_estimators=n_estimators,
                      min_samples_leaf=min_samples_leaf, cv=cv, seed=random_state,
                      max_samples=max_samples, model_y=repr(model_y), model_t=repr(model_t),
                      forest="CausalForestDML(defaults)")
    if cache_name and not force_refit:
        cached = load_cache(cache_name, fp)
        if cached is not None:
            return cached["model"], cached["results"], cached["cate"]

    # Optional subsampling for computational feasibility (whole traders)
    n = len(Y)
    rng = np.random.RandomState(random_state)
    groups_fit = groups
    if max_samples and n > max_samples:
        if groups is not None:
            uniq = np.unique(groups)
            rows_per_group = n / len(uniq)
            n_groups = max(1, int(max_samples / rows_per_group))
            keep = rng.choice(uniq, size=min(n_groups, len(uniq)), replace=False)
            idx = np.flatnonzero(np.isin(groups, keep))
            groups_fit = groups[idx]
        else:
            idx = rng.choice(n, size=max_samples, replace=False)
        logger.info(
            f"Subsampling from {n:,} to {len(idx):,} rows for CausalForest"
            + (" (by trader)" if groups is not None else "")
        )
        Y_fit, T_fit = Y[idx], T[idx]
        X_fit, W_fit = X[idx], W[idx]
    else:
        idx = np.arange(n)
        Y_fit, T_fit, X_fit, W_fit = Y, T, X, W

    logger.info(
        f"Fitting CausalForestDML: n_estimators={n_estimators}, "
        f"min_samples_leaf={min_samples_leaf}, cv={cv}, "
        f"N_fit={len(Y_fit):,}, X_cols={X_fit.shape[1]}, W_cols={W_fit.shape[1]}"
    )

    cf = CausalForestDML(
        model_y=model_y,
        model_t=model_t,
        n_estimators=n_estimators,
        min_samples_leaf=min_samples_leaf,
        cv=trader_folds(groups_fit, cv, random_state) if groups_fit is not None else cv,
        random_state=random_state,
        discrete_treatment=False,
    )

    cf.fit(Y=Y_fit, T=T_fit, X=X_fit, W=W_fit)

    # Predict CATE on all rows passed in (even if we subsampled for fitting)
    cate = cf.effect(X).ravel()

    # Summary stats
    results = {
        "fit_row_index": idx,
        "n_fit_traders": int(len(np.unique(groups_fit))) if groups_fit is not None else None,
        "ate_from_cate": float(np.mean(cate)),
        "cate_mean": float(np.mean(cate)),
        "cate_median": float(np.median(cate)),
        "cate_std": float(np.std(cate)),
        "cate_min": float(np.min(cate)),
        "cate_max": float(np.max(cate)),
        "share_positive": float(np.mean(cate > 0)),
        "share_negative": float(np.mean(cate < 0)),
        "n_fit": int(len(Y_fit)),
        "n_predict": int(len(cate)),
        "n_estimators": n_estimators,
        "model_name": "CausalForestDML",
    }

    # Feature importances
    try:
        importances = cf.feature_importances_
        results["feature_importances"] = importances.tolist()
    except Exception:
        logger.warning("Could not extract feature importances")

    logger.info(
        f"CausalForestDML: ATE(from CATE)={results['ate_from_cate']:.4f}, "
        f"CATE range=[{results['cate_min']:.4f}, {results['cate_max']:.4f}], "
        f"share>0={results['share_positive']:.1%}"
    )

    # Cache
    if cache_name:
        save_cache(
            f"{cache_name}_{fp}",
            {"model": cf, "results": results, "cate": cate},
        )

    return cf, results, cate


def wald_psd(fit, R: np.ndarray) -> Dict[str, float]:
    """
    Wald test of R @ params = 0 that stays valid when the covariance matrix is
    not positive semi-definite, as two-way cluster-robust matrices can be:
    negative eigenvalues are set to zero (Cameron, Gelbach & Miller 2011) and
    the statistic is chi-squared with df = number of positive eigenvalues.
    """
    from scipy import stats

    b = R @ np.asarray(fit.params)
    V = R @ np.asarray(fit.cov_params()) @ R.T
    vals, vecs = np.linalg.eigh(V)
    keep = vals > 1e-12 * max(vals.max(), 1e-300)
    proj = vecs[:, keep].T @ b
    chi2 = float(np.sum(proj ** 2 / vals[keep]))
    df = int(keep.sum())
    return {"stat": chi2, "df": df, "pvalue": float(stats.chi2.sf(chi2, df)),
            "n_negative_eigenvalues": int((vals < 0).sum())}


def group_effects(
    labels: np.ndarray,
    Y_res: np.ndarray,
    T_res: np.ndarray,
    groups: np.ndarray,
    order: Optional[List[Any]] = None,
    time_ids: Optional[np.ndarray] = None,
) -> Dict[str, Any]:
    """
    Average treatment effect within each label group, from cross-fitted DML
    residuals: OLS of Y_res on T_res x group dummies (no intercept), with
    trader-clustered SEs, plus a Wald test that all group effects are equal.
    With `time_ids` (and enough week clusters) the Wald test is also reported
    with two-way (trader x week) clustered SEs.

    Valid as a heterogeneity test when the labels do not depend on the
    outcomes of the same rows (e.g. observed characteristics, or CATE
    quantiles from a forest fit on OTHER traders).
    """
    import statsmodels.api as sm

    labels = np.asarray(labels)
    if order is None:
        order = sorted(pd.unique(labels).tolist())
    order = [g for g in order if (labels == g).any()]
    k = len(order)
    D = np.column_stack([(labels == g).astype(float) for g in order])
    Z = D * np.asarray(T_res).reshape(-1, 1)

    fit = sm.OLS(np.asarray(Y_res).ravel(), Z).fit(
        cov_type="cluster", cov_kwds={"groups": pd.factorize(groups)[0]}
    )
    ci = fit.conf_int(alpha=0.05)
    out_groups = [{
        "group": g if not isinstance(g, np.generic) else g.item(),
        "effect": float(fit.params[j]),
        "se": float(fit.bse[j]),
        "ci_lower": float(ci[j, 0]),
        "ci_upper": float(ci[j, 1]),
        "n": int(D[:, j].sum()),
        "treatment_weight": float((Z[:, j] ** 2).sum() / (Z ** 2).sum()),
    } for j, g in enumerate(order)]

    res = {"groups": out_groups, "n": int(len(labels))}
    if k > 1:
        R = np.zeros((k - 1, k))
        for j in range(k - 1):
            R[j, j], R[j, j + 1] = 1.0, -1.0
        wald = fit.wald_test(R, scalar=True)
        diff = fit.t_test(np.eye(k)[k - 1] - np.eye(k)[0])
        res.update({
            "wald_equal_stat": float(wald.statistic),
            "wald_equal_pvalue": float(wald.pvalue),
            "last_minus_first": float(np.asarray(diff.effect).ravel()[0]),
            "last_minus_first_pvalue": float(np.asarray(diff.pvalue).ravel()[0]),
        })
        if time_ids is not None and len(np.unique(time_ids)) >= MIN_CLUSTERS_FOR_CRVE:
            two = np.column_stack([pd.factorize(groups)[0], pd.factorize(time_ids)[0]])
            fit2 = sm.OLS(np.asarray(Y_res).ravel(), Z).fit(
                cov_type="cluster", cov_kwds={"groups": two})
            res["wald_equal_pvalue_twoway"] = wald_psd(fit2, R)["pvalue"]
    return res


def blp_calibration(
    cate: np.ndarray,
    Y_res: np.ndarray,
    T_res: np.ndarray,
    groups: np.ndarray,
    time_ids: Optional[np.ndarray] = None,
) -> Dict[str, Any]:
    """
    Best linear predictor (BLP) calibration test on held-out data
    (Chernozhukov, Demirer, Duflo & Fernandez-Val 2025):
        Y_res = b1 * T_res + b2 * T_res * (S - mean S) + e,
    with S the predicted CATE from the forest. b2 = 1: perfectly calibrated
    predictions; b2 = 0: the predictions carry no information about
    heterogeneity. Trader-clustered SEs (two-way if enough week clusters).
    """
    import statsmodels.api as sm

    S = np.asarray(cate, dtype=np.float64).ravel()
    T_res = np.asarray(T_res, dtype=np.float64).ravel()
    Z = np.column_stack([T_res, T_res * (S - S.mean())])
    y = np.asarray(Y_res, dtype=np.float64).ravel()
    fit = sm.OLS(y, Z).fit(cov_type="cluster", cov_kwds={"groups": pd.factorize(groups)[0]})
    out = {"b1_ate": float(fit.params[0]), "b1_se": float(fit.bse[0]),
           "b2_het": float(fit.params[1]), "b2_se": float(fit.bse[1]),
           "b2_pvalue": float(fit.pvalues[1]), "n": int(len(y))}
    if time_ids is not None and len(np.unique(time_ids)) >= MIN_CLUSTERS_FOR_CRVE:
        two = np.column_stack([pd.factorize(groups)[0], pd.factorize(time_ids)[0]])
        fit2 = sm.OLS(y, Z).fit(cov_type="cluster", cov_kwds={"groups": two})
        out.update({"b2_se_twoway": float(fit2.bse[1]),
                    "b2_pvalue_twoway": float(fit2.pvalues[1])})
    return out


def blp_moderators(
    moderators: pd.DataFrame,
    blocks: np.ndarray,
    Y_res: np.ndarray,
    T_res: np.ndarray,
    groups: np.ndarray,
    time_ids: Optional[np.ndarray] = None,
) -> Dict[str, Any]:
    """
    Linear projection of the treatment effect on trader-level moderators,
    with a separate intercept for each period block (Semenova & Chernozhukov
    2021):
        Y_res = sum_b theta_b * T_res * 1[block = b] + T_res * (X_std' beta) + e.
    beta measures trader heterogeneity net of calendar (block) differences.
    Moderators are standardised, so beta is the change in the effect per SD.
    Joint Wald test of beta = 0 with trader-clustered (and two-way) SEs.
    """
    import statsmodels.api as sm

    T_res = np.asarray(T_res, dtype=np.float64).ravel()
    y = np.asarray(Y_res, dtype=np.float64).ravel()
    blocks = np.asarray(blocks)
    block_names = list(pd.unique(blocks))
    Xs = (moderators - moderators.mean()) / moderators.std(ddof=0)
    Xs = Xs.fillna(0.0)
    Z = np.column_stack([T_res * (blocks == b) for b in block_names]
                        + [T_res * Xs[c].to_numpy() for c in Xs.columns])
    names = [f"block: {b}" for b in block_names] + list(Xs.columns)
    k_b = len(block_names)
    R = np.zeros((Xs.shape[1], Z.shape[1]))
    R[np.arange(Xs.shape[1]), k_b + np.arange(Xs.shape[1])] = 1.0

    fit = sm.OLS(y, Z).fit(cov_type="cluster", cov_kwds={"groups": pd.factorize(groups)[0]})
    coefs = [{"term": n, "coef": float(fit.params[j]), "se": float(fit.bse[j]),
              "pvalue": float(fit.pvalues[j])} for j, n in enumerate(names)]
    est_cov = np.asarray(fit.cov_params())[k_b:, k_b:]
    sd = np.sqrt(np.diag(est_cov))
    out = {"terms": coefs,
           "joint_moderators_pvalue": wald_psd(fit, R)["pvalue"],
           "moderator_estimate_correlation": (est_cov / np.outer(sd, sd)).round(3).tolist(),
           "n": int(len(y))}
    if time_ids is not None and len(np.unique(time_ids)) >= MIN_CLUSTERS_FOR_CRVE:
        two = np.column_stack([pd.factorize(groups)[0], pd.factorize(time_ids)[0]])
        fit2 = sm.OLS(y, Z).fit(cov_type="cluster", cov_kwds={"groups": two})
        for j, c in enumerate(coefs):
            c["se_twoway"], c["pvalue_twoway"] = float(fit2.bse[j]), float(fit2.pvalues[j])
        out["joint_moderators_pvalue_twoway"] = wald_psd(fit2, R)["pvalue"]
    return out


def gates_test(
    cate: np.ndarray,
    Y_res: np.ndarray,
    T_res: np.ndarray,
    groups: np.ndarray,
    n_bins: int = 4,
    block: Optional[np.ndarray] = None,
    time_ids: Optional[np.ndarray] = None,
) -> Dict[str, Any]:
    """
    Group Average Treatment Effects (GATES) by CATE quantile, on HELD-OUT data.

    `cate` must be predicted for observations NOT used to fit the forest, and
    `Y_res`/`T_res` must be cross-fitted DML residuals for the same rows.
    Rejecting equality of group effects is evidence of real heterogeneity;
    the forest's point estimates alone are not.

    With `block` (e.g. calendar-week block labels), CATE quantiles are formed
    and tested within each block as well, to check that the gradient is not
    just between-block (calendar) variation.
    """
    cate = np.asarray(cate).ravel()
    bins = pd.qcut(cate, q=n_bins, labels=False, duplicates="drop")
    res = group_effects(bins, Y_res, T_res, groups, time_ids=time_ids)
    for g in res["groups"]:
        g["cate_mean"] = float(cate[bins == g["group"]].mean())
        g["group"] = int(g["group"]) + 1
        g["gate"] = g.pop("effect")
    res["top_minus_bottom"] = res.pop("last_minus_first")
    res["top_minus_bottom_pvalue"] = res.pop("last_minus_first_pvalue")

    if block is not None:
        block = np.asarray(block)
        res["by_block"] = {}
        for b in pd.unique(block):
            m = block == b
            if m.sum() < 100 * n_bins:
                continue
            sub = gates_test(cate[m], np.asarray(Y_res)[m], np.asarray(T_res)[m],
                             np.asarray(groups)[m], n_bins=n_bins,
                             time_ids=None if time_ids is None else np.asarray(time_ids)[m])
            if isinstance(b, np.generic):
                b = b.item()
            res["by_block"][str(b)] = sub
    return res
