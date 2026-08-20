"""
Causal models for estimating the effect of self-trading on gains.

Implements:
- naive_ols(): OLS of Y on T with no controls (confounded baseline)
- fit_linear_dml(): LinearDML for ATE with various nuisance learners
- fit_causal_forest(): CausalForestDML for CATE heterogeneity
- Model caching to avoid refitting on expensive ~1M-row panel
"""

import logging
import os
import pickle
from typing import Dict, Any, Optional, Tuple, List

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

# Cache directory for fitted models
CACHE_DIR = "data/processed"


def naive_ols(
    Y: np.ndarray,
    T: np.ndarray,
    add_constant: bool = True,
) -> Dict[str, Any]:
    """
    Naive OLS regression of Y on T with NO controls.

    This is the confounded baseline — reported side-by-side with DML
    to demonstrate the importance of causal identification.

    Uses heteroskedasticity-robust (HC1) standard errors.

    Parameters
    ----------
    Y : array-like, shape (n,)
        Outcome variable.
    T : array-like, shape (n,)
        Treatment variable.
    add_constant : bool
        Whether to add an intercept.

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
    model = sm.OLS(Y, X).fit(cov_type="HC1")

    idx = 1 if add_constant else 0
    result = {
        "coef": float(model.params[idx]),
        "se": float(model.bse[idx]),
        "ci_lower": float(model.conf_int(alpha=0.05)[idx, 0]),
        "ci_upper": float(model.conf_int(alpha=0.05)[idx, 1]),
        "pvalue": float(model.pvalues[idx]),
        "r2": float(model.rsquared),
        "n": int(len(Y)),
        "model_name": "Naive OLS (no controls)",
    }

    logger.info(
        f"Naive OLS: coef={result['coef']:.4f}, SE={result['se']:.4f}, "
        f"95% CI=[{result['ci_lower']:.4f}, {result['ci_upper']:.4f}], "
        f"p={result['pvalue']:.4g}, N={result['n']:,}"
    )

    return result


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
) -> Tuple[Any, Dict[str, Any]]:
    """
    Fit LinearDML for the Average Treatment Effect (ATE).

    Parameters
    ----------
    Y : array-like, shape (n,)
        Outcome.
    T : array-like, shape (n,)
        Treatment (continuous: total_self_trxn or self_trade_ratio).
    W : array-like, shape (n, p)
        Confounders (nuisance-only controls).
    X : array-like, shape (n, q), optional
        Effect modifiers. None = ATE only.
    nuisance_learner : str
        One of 'histgbr', 'lightgbm', 'lasso', 'ridge'.
    cv : int
        Number of cross-fitting folds.
    random_state : int
        Random seed.
    cache_name : str, optional
        If provided, cache the fitted model.
    force_refit : bool
        If True, refit even if cache exists.

    Returns
    -------
    Tuple[LinearDML, Dict]
        Fitted model and results dict.
    """
    from econml.dml import LinearDML

    # Check cache
    if cache_name and not force_refit:
        cache_path = os.path.join(CACHE_DIR, f"{cache_name}.pkl")
        if os.path.exists(cache_path):
            logger.info(f"Loading cached LinearDML from {cache_path}")
            with open(cache_path, "rb") as f:
                cached = pickle.load(f)
            return cached["model"], cached["results"]

    # Build nuisance learners
    model_y, model_t = _build_nuisance_learners(nuisance_learner, random_state)

    logger.info(
        f"Fitting LinearDML: nuisance={nuisance_learner}, cv={cv}, "
        f"N={len(Y):,}, W_cols={W.shape[1]}"
    )

    dml = LinearDML(
        model_y=model_y,
        model_t=model_t,
        cv=cv,
        random_state=random_state,
        discrete_treatment=False,
    )

    dml.fit(Y=Y, T=T, W=W, X=X)

    # Extract results
    ate = dml.ate(X=X)
    ate_inf = dml.ate_inference(X=X)
    ci = ate_inf.conf_int_mean()

    results = {
        "ate": float(ate),
        "ate_se": float(ate_inf.stderr_mean),
        "ate_ci_lower": float(ci[0]),
        "ate_ci_upper": float(ci[1]),
        "ate_pvalue": float(ate_inf.pvalue()),
        "n": int(len(Y)),
        "nuisance_learner": nuisance_learner,
        "cv": cv,
        "model_name": f"LinearDML ({nuisance_learner})",
    }

    logger.info(
        f"LinearDML ATE: {results['ate']:.4f} "
        f"(SE={results['ate_se']:.4f}, "
        f"95% CI=[{results['ate_ci_lower']:.4f}, {results['ate_ci_upper']:.4f}], "
        f"p={results['ate_pvalue']:.4g})"
    )

    # Cache
    if cache_name:
        _cache_result(cache_name, {"model": dml, "results": results})

    return dml, results


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
) -> Tuple[Any, Dict[str, Any], np.ndarray]:
    """
    Fit CausalForestDML for treatment-effect heterogeneity (CATE).

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

    # Check cache
    if cache_name and not force_refit:
        cache_path = os.path.join(CACHE_DIR, f"{cache_name}.pkl")
        if os.path.exists(cache_path):
            logger.info(f"Loading cached CausalForestDML from {cache_path}")
            with open(cache_path, "rb") as f:
                cached = pickle.load(f)
            return cached["model"], cached["results"], cached["cate"]

    # Optional subsampling for computational feasibility
    n = len(Y)
    if max_samples and n > max_samples:
        logger.info(
            f"Subsampling from {n:,} to {max_samples:,} rows for CausalForest"
        )
        rng = np.random.RandomState(random_state)
        idx = rng.choice(n, size=max_samples, replace=False)
        Y_fit, T_fit = Y[idx], T[idx]
        X_fit, W_fit = X[idx], W[idx]
    else:
        Y_fit, T_fit, X_fit, W_fit = Y, T, X, W

    # Build nuisance learners (always HistGBR for forest)
    model_y, model_t = _build_nuisance_learners("histgbr", random_state)

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
        cv=cv,
        random_state=random_state,
        discrete_treatment=False,
    )

    cf.fit(Y=Y_fit, T=T_fit, X=X_fit, W=W_fit)

    # Predict CATE on FULL sample (even if we subsampled for fitting)
    cate = cf.effect(X).ravel()

    # Summary stats
    results = {
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
        _cache_result(
            cache_name,
            {"model": cf, "results": results, "cate": cate},
        )

    return cf, results, cate


def _build_nuisance_learners(
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

        model_y = LassoCV(cv=5, random_state=random_state, max_iter=5000)
        model_t = LassoCV(cv=5, random_state=random_state, max_iter=5000)

    elif learner_type == "ridge":
        from sklearn.linear_model import RidgeCV

        model_y = RidgeCV(cv=5)
        model_t = RidgeCV(cv=5)

    else:
        raise ValueError(f"Unknown learner type: {learner_type}")

    return model_y, model_t


def _cache_result(cache_name: str, obj: dict) -> None:
    """Save a fitted model/result to the cache directory."""
    cache_path = os.path.join(CACHE_DIR, f"{cache_name}.pkl")
    os.makedirs(CACHE_DIR, exist_ok=True)
    with open(cache_path, "wb") as f:
        pickle.dump(obj, f)
    logger.info(f"Cached to {cache_path}")


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
