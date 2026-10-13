
"""
Double machine learning estimators for the effect of self-trading on gains.

- naive_ols(): OLS of Y on T with no controls (confounded baseline)
- linear_fe_ols(): linear two-way FE regression (Mani et al.-style benchmark)
- fit_linear_dml(): LinearDML (partially linear model) with cross-fitting
  folds that never split a trader (trader_folds) and trader-clustered and
  two-way (trader x week) clustered SEs (clustered_final_stage)
- build_nuisance_learners(), prepare_model_arrays(): helpers
"""

import logging
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from social_trading_dml.config import MIN_CLUSTERS_FOR_CRVE, WILD_BOOTSTRAP_REPS
from social_trading_dml.estimation.cache import fingerprint, load_cache, save_cache
from social_trading_dml.estimation.inference import partial_out, wild_cluster_bootstrap

logger = logging.getLogger(__name__)


def clustered_final_stage(
    Y_res: np.ndarray,
    T_res: np.ndarray,
    groups: Optional[np.ndarray] = None,
    time_ids: Optional[np.ndarray] = None,
) -> Dict[str, float]:
    """
    Final-stage OLS of Y residuals on T residuals (no intercept), the DML
    partially-linear estimator, with trader-clustered (or HC1) standard errors.

    With `time_ids` as well, also reports week-level inference, which matters
    here because the identifying variation is concentrated in a few weeks:
    - a wild cluster bootstrap by week (WCR p-value and the CI from inverting it;
      see estimation/inference.py) and the effective number of weeks;
    - two-way (trader x week) cluster-robust SEs (Cameron, Gelbach & Miller
      2011).
    With fewer than MIN_CLUSTERS_FOR_CRVE week clusters these are reported as
    None, because cluster-robust inference is not reliable there.

    Also returns the t-statistic, residual degrees of freedom and the partial
    R^2 of T with Y, which feed the Cinelli-Hazlett sensitivity analysis.
    """
    import statsmodels.api as sm

    Y_res = np.asarray(Y_res, dtype=np.float64).ravel()
    T_res = np.asarray(T_res, dtype=np.float64).ravel()
    if groups is not None:
        fit = sm.OLS(Y_res, T_res).fit(
            cov_type="cluster", cov_kwds={"groups": pd.factorize(groups)[0]}
        )
    else:
        fit = sm.OLS(Y_res, T_res).fit(cov_type="HC1")

    coef, se = float(fit.params[0]), float(fit.bse[0])
    dof = int(len(Y_res) - 1)
    ci = fit.conf_int(alpha=0.05)[0]
    # Cinelli & Hazlett (2020) quantities use the classical OLS t-statistic
    t_classical = float(sm.OLS(Y_res, T_res).fit().tvalues[0])
    out = {
        "ate": coef,
        "ate_se": se,
        "ate_ci_lower": float(ci[0]),
        "ate_ci_upper": float(ci[1]),
        "ate_pvalue": float(fit.pvalues[0]),
        "t_stat": float(coef / se),
        "t_stat_classical": t_classical,
        "dof": dof,
        "partial_r2_ty": float(t_classical ** 2 / (t_classical ** 2 + dof)),
        "n_clusters": int(len(np.unique(groups))) if groups is not None else None,
        "se_type": "cluster(trader)" if groups is not None else "HC1",
    }
    if time_ids is not None:
        out.update(week_inference(Y_res, T_res, groups, time_ids))
    return out


def week_inference(
    Y_res: np.ndarray,
    T_res: np.ndarray,
    groups: Optional[np.ndarray],
    time_ids: np.ndarray,
) -> Dict[str, Any]:
    """Week-level inference for a partialled-out coefficient (see clustered_final_stage)."""
    import statsmodels.api as sm

    n_weeks = int(len(np.unique(time_ids)))
    boot = wild_cluster_bootstrap(Y_res, T_res, time_ids, reps=WILD_BOOTSTRAP_REPS)
    out = {"n_time_clusters": n_weeks,
           "effective_time_clusters": boot["effective_clusters"]}
    keys = ["ate_se_week", "ate_pvalue_week_wild", "ate_ci_lower_week_wild",
            "ate_ci_upper_week_wild", "ate_se_twoway", "ate_ci_lower_twoway",
            "ate_ci_upper_twoway", "ate_pvalue_twoway"]
    if n_weeks < MIN_CLUSTERS_FOR_CRVE:
        out.update({k: None for k in keys})
        return out
    out.update({
        "ate_se_week": boot["se_cluster"],
        "ate_pvalue_week_wild": boot["p_wild"],
        "ate_ci_lower_week_wild": boot["ci_lower_wild"],
        "ate_ci_upper_week_wild": boot["ci_upper_wild"],
    })
    if groups is not None:
        two = np.column_stack([pd.factorize(groups)[0], pd.factorize(time_ids)[0]])
        fit2 = sm.OLS(Y_res, T_res).fit(cov_type="cluster", cov_kwds={"groups": two})
        ci2 = fit2.conf_int(alpha=0.05)[0]
        out.update({
            "ate_se_twoway": float(fit2.bse[0]),
            "ate_ci_lower_twoway": float(ci2[0]),
            "ate_ci_upper_twoway": float(ci2[1]),
            "ate_pvalue_twoway": float(fit2.pvalues[0]),
        })
    return out


def trader_folds(groups: np.ndarray, n_folds: int = 5,
                 random_state: int = 42) -> List[Tuple[np.ndarray, np.ndarray]]:
    """
    Cross-fitting folds that never split a trader: traders are shuffled with a
    fixed seed and assigned round-robin to folds.
    """
    codes, uniq = pd.factorize(np.asarray(groups))
    rng = np.random.RandomState(random_state)
    fold_of_group = np.empty(len(uniq), dtype=int)
    fold_of_group[rng.permutation(len(uniq))] = np.arange(len(uniq)) % n_folds
    fold = fold_of_group[codes]
    idx = np.arange(len(codes))
    return [(idx[fold != k], idx[fold == k]) for k in range(n_folds)]


def naive_ols(
    Y: np.ndarray,
    T: np.ndarray,
    add_constant: bool = True,
    groups: Optional[np.ndarray] = None,
) -> Dict[str, Any]:
    """
    Naive OLS regression of Y on T with NO controls.

    This is the confounded baseline — reported side-by-side with DML
    to demonstrate the importance of causal identification.

    Uses trader-clustered SEs when `groups` is given, else HC1.

    Parameters
    ----------
    Y : array-like, shape (n,)
        Outcome variable.
    T : array-like, shape (n,)
        Treatment variable.
    add_constant : bool
        Whether to add an intercept.
    groups : array-like, shape (n,), optional
        Cluster ids (trader_id).

    Returns
    -------
    Dict with keys: coef, se, ci_lower, ci_upper, pvalue, r2, n
    """
    import statsmodels.api as sm

    Y = np.asarray(Y, dtype=np.float64).ravel()
    T = np.asarray(T, dtype=np.float64).ravel()

    # Drop any NaN/Inf
    valid = np.isfinite(Y) & np.isfinite(T)
    Y, T = Y[valid], T[valid]

    X = sm.add_constant(T) if add_constant else T.reshape(-1, 1)
    if groups is not None:
        g = pd.factorize(np.asarray(groups)[valid])[0]
        model = sm.OLS(Y, X).fit(cov_type="cluster", cov_kwds={"groups": g})
    else:
        model = sm.OLS(Y, X).fit(cov_type="HC1")

    idx = 1 if add_constant else 0
    result = {
        "coef": float(model.params[idx]),
        "se": float(model.bse[idx]),
        "ci_lower": float(model.conf_int(alpha=0.05)[idx, 0]),
        "ci_upper": float(model.conf_int(alpha=0.05)[idx, 1]),
        "pvalue": float(model.pvalues[idx]),
        "ate_pvalue": float(model.pvalues[idx]),
        "r2": float(model.rsquared),
        "n": int(len(Y)),
        "model_name": "Naive OLS (no controls)",
        "se_type": "cluster(trader)" if groups is not None else "HC1",
    }

    logger.info(
        f"Naive OLS: coef={result['coef']:.4f}, SE={result['se']:.4f}, "
        f"95% CI=[{result['ci_lower']:.4f}, {result['ci_upper']:.4f}], "
        f"p={result['pvalue']:.4g}, N={result['n']:,}"
    )

    return result


def linear_fe_ols(arrays: Dict[str, Any]) -> Dict[str, Any]:
    """
    Linear two-way FE regression of Y on T and the controls (Mani et al.-style
    benchmark). Computed by Frisch-Waugh-Lovell partialling, so it gets exactly
    the same inference as the DML final stage (trader-clustered, week wild
    bootstrap, two-way).
    """
    W = np.asarray(arrays["W"], dtype=np.float64)
    T_res = partial_out(arrays["T"], W)
    Y_res = partial_out(arrays["Y"], W)
    res = clustered_final_stage(Y_res, T_res, arrays["groups"], arrays.get("periods"))
    res.update({"model_name": "Linear two-way FE OLS (clustered)", "n": int(len(Y_res))})
    return res


def fit_linear_dml(
    Y: np.ndarray,
    T: np.ndarray,
    W: np.ndarray,
    X: Optional[np.ndarray] = None,
    nuisance_learner: str = "histgbr",
    cv: int = 5,
    random_state: int = 42,
    cache_name: Optional[str] = None,
    force_refit: bool = False,
    groups: Optional[np.ndarray] = None,
    return_residuals: bool = False,
    time_ids: Optional[np.ndarray] = None,
) -> Tuple[Any, Dict[str, Any]]:
    """
    Fit LinearDML for the Average Treatment Effect (ATE).

    With `groups` (trader ids), cross-fitting uses trader_folds() so a
    trader's whole history sits in one fold, and the reported SE/CI are
    clustered by trader (final-stage OLS on the cross-fitted residuals).
    With `time_ids` (weeks) as well, two-way trader x week clustered SEs are
    added. Without `groups`, SEs are HC1.

    The cache stores the model and cross-fitted residuals; the summary is
    recomputed from the residuals on every load.

    Parameters
    ----------
    Y : array-like, shape (n,)
        Outcome.
    T : array-like, shape (n,)
        Treatment (continuous: total_self_trxn or self_trade_ratio).
    W : array-like, shape (n, p)
        Confounders (nuisance-only controls).
    X : array-like, shape (n, q), optional
        Effect modifiers. None = ATE only (required for clustered SEs).
    nuisance_learner : str
        One of 'histgbr', 'lightgbm', 'lasso', 'ridge'.
    cv : int
        Number of cross-fitting folds.
    random_state : int
        Random seed.
    cache_name : str, optional
        If provided, cache the fitted model (file name includes a data hash).
    force_refit : bool
        If True, refit even if cache exists.
    groups : array-like, shape (n,), optional
        Cluster ids (trader_id) for grouped cross-fitting and clustered SEs.
    return_residuals : bool
        If True, results include 'Y_res' and 'T_res' arrays (not cached
        in the summary JSON).

    Returns
    -------
    Tuple[LinearDML, Dict]
        Fitted model and results dict.
    """
    from econml.dml import LinearDML

    model_y, model_t = build_nuisance_learners(nuisance_learner, random_state)
    fp = fingerprint(Y, T, W, X, groups, learner=nuisance_learner, cv=cv,
                      seed=random_state, model_y=repr(model_y), model_t=repr(model_t))

    def summarize(dml, Y_res, T_res):
        if X is None:
            res = clustered_final_stage(Y_res, T_res, groups, time_ids)
        else:
            ate_inf = dml.ate_inference(X=X)
            ci = ate_inf.conf_int_mean()
            res = {
                "ate": float(dml.ate(X=X)),
                "ate_se": float(ate_inf.stderr_mean),
                "ate_ci_lower": float(ci[0]),
                "ate_ci_upper": float(ci[1]),
                "ate_pvalue": float(ate_inf.pvalue()),
                "se_type": "econml (not clustered)",
            }
        res.update({
            "n": int(len(Y)),
            "nuisance_learner": nuisance_learner,
            "cv": cv,
            "model_name": f"LinearDML ({nuisance_learner})",
            "nuisance_r2_y": float(1 - np.var(Y_res) / np.var(Y)) if np.var(Y) > 0 else np.nan,
            "nuisance_r2_t": float(1 - np.var(T_res) / np.var(T)) if np.var(T) > 0 else np.nan,
        })
        if return_residuals:
            res["Y_res"], res["T_res"] = Y_res, T_res
        return res

    if cache_name and not force_refit:
        cached = load_cache(cache_name, fp)
        if cached is not None:
            Y_res, T_res = cached["residuals"]
            return cached["model"], summarize(cached["model"], Y_res, T_res)

    logger.info(
        f"Fitting LinearDML: nuisance={nuisance_learner}, cv={cv}, "
        f"N={len(Y):,}, W_cols={W.shape[1]}, grouped={groups is not None}"
    )

    dml = LinearDML(
        model_y=model_y,
        model_t=model_t,
        cv=trader_folds(groups, cv, random_state) if groups is not None else cv,
        random_state=random_state,
        discrete_treatment=False,
    )

    dml.fit(Y=Y, T=T, W=W, X=X, cache_values=True)

    Y_res, T_res = dml.residuals_[0], dml.residuals_[1]
    Y_res, T_res = np.asarray(Y_res).ravel(), np.asarray(T_res).ravel()

    results = summarize(dml, Y_res, T_res)
    if X is None:
        econml_ate = float(dml.ate())
        if not np.isclose(econml_ate, results["ate"], rtol=1e-6, atol=1e-8):
            logger.warning(
                f"Final-stage OLS ({results['ate']:.6f}) differs from econml "
                f"ATE ({econml_ate:.6f})"
            )

    logger.info(
        f"LinearDML ATE: {results['ate']:.4f} "
        f"(SE={results['ate_se']:.4f} [{results['se_type']}], "
        f"95% CI=[{results['ate_ci_lower']:.4f}, {results['ate_ci_upper']:.4f}], "
        f"p={results['ate_pvalue']:.4g})"
    )

    if cache_name:
        save_cache(
            f"{cache_name}_{fp}",
            {"model": dml, "residuals": (Y_res, T_res)},
        )

    return dml, results


def build_nuisance_learners(
    learner_type: str,
    random_state: int = 42,
):
    """
    Build nuisance model pair (model_y, model_t) for DML.

    Parameters
    ----------
    learner_type : str
        One of 'histgbr', 'lightgbm', 'lasso', 'ridge'.
    random_state : int
        Random seed.

    Returns
    -------
    Tuple[estimator, estimator]
        (model_y, model_t)
    """
    if learner_type == "histgbr":
        from sklearn.ensemble import HistGradientBoostingRegressor

        model_y = HistGradientBoostingRegressor(
            max_iter=200, max_depth=6, learning_rate=0.1,
            random_state=random_state,
        )
        model_t = HistGradientBoostingRegressor(
            max_iter=200, max_depth=6, learning_rate=0.1,
            random_state=random_state,
        )

    elif learner_type == "lightgbm":
        import lightgbm as lgb

        model_y = lgb.LGBMRegressor(
            n_estimators=200, max_depth=6, learning_rate=0.1,
            random_state=random_state, verbose=-1,
        )
        model_t = lgb.LGBMRegressor(
            n_estimators=200, max_depth=6, learning_rate=0.1,
            random_state=random_state, verbose=-1,
        )

    elif learner_type == "lasso":
        from sklearn.linear_model import LassoCV
        from sklearn.pipeline import make_pipeline
        from sklearn.preprocessing import StandardScaler

        model_y = make_pipeline(
            StandardScaler(), LassoCV(cv=5, random_state=random_state, max_iter=5000)
        )
        model_t = make_pipeline(
            StandardScaler(), LassoCV(cv=5, random_state=random_state, max_iter=5000)
        )

    elif learner_type == "ridge":
        from sklearn.linear_model import RidgeCV
        from sklearn.pipeline import make_pipeline
        from sklearn.preprocessing import StandardScaler

        model_y = make_pipeline(StandardScaler(), RidgeCV())
        model_t = make_pipeline(StandardScaler(), RidgeCV())

    else:
        raise ValueError(f"Unknown learner type: {learner_type}")

    return model_y, model_t


def prepare_model_arrays(
    df: pd.DataFrame,
    treatment_col: str = "total_self_trxn",
    outcome_col: str = "total_gain_w",
    confounder_cols: Optional[List[str]] = None,
    heterogeneity_cols: Optional[List[str]] = None,
) -> Dict[str, Any]:
    """
    Prepare numpy arrays for model fitting from a DataFrame.

    Parameters
    ----------
    df : pd.DataFrame
        Analysis-ready DataFrame.
    treatment_col : str
        Treatment column name.
    outcome_col : str
        Outcome column name.
    confounder_cols : list of str
        All confounder columns (W for DML).
    heterogeneity_cols : list of str, optional
        Subset for X (heterogeneity features). If None, X = None.

    Returns
    -------
    Dict with keys: Y, T, W, X, feature_names_W, feature_names_X
    """
    Y = df[outcome_col].values.astype(np.float64)
    T = df[treatment_col].values.astype(np.float64)

    if confounder_cols is None:
        raise ValueError("Must specify confounder_cols")

    # Verify all columns exist
    missing = [c for c in confounder_cols if c not in df.columns]
    if missing:
        raise ValueError(f"Confounder columns not in DataFrame: {missing}")

    W = df[confounder_cols].values.astype(np.float64)

    X = None
    x_names = None
    if heterogeneity_cols:
        missing_x = [c for c in heterogeneity_cols if c not in df.columns]
        if missing_x:
            raise ValueError(f"Heterogeneity columns not in DataFrame: {missing_x}")
        X = df[heterogeneity_cols].values.astype(np.float64)
        x_names = heterogeneity_cols

    # Sanity checks
    assert np.isfinite(Y).all(), f"Y contains {(~np.isfinite(Y)).sum()} non-finite values"
    assert np.isfinite(T).all(), f"T contains {(~np.isfinite(T)).sum()} non-finite values"
    assert np.isfinite(W).all(), f"W contains {(~np.isfinite(W)).sum()} non-finite values"
    if X is not None:
        assert np.isfinite(X).all(), f"X contains {(~np.isfinite(X)).sum()} non-finite values"

    logger.info(
        f"Model arrays: Y({Y.shape}), T({T.shape}), "
        f"W({W.shape}), X({X.shape if X is not None else None})"
    )

    return {
        "Y": Y,
        "T": T,
        "W": W,
        "X": X,
        "feature_names_W": confounder_cols,
        "feature_names_X": x_names,
    }
