"""
Robustness suite for the causal ML analysis.

Implements:
- Specification grid: {count vs. ratio treatment} x {HistGBR vs. LightGBM vs. linear}
  x {full panel vs. tenure >= 23 weeks} x {with vs. without country controls}
  x {keep_country primary vs. alternate}
- Overlap/positivity check
- Outlier exclusion (>3 SD)
- Heckman-style selection correction for active-weeks restriction
- DoWhy refutation tests
"""

import logging
import itertools
from typing import Dict, Any, List, Optional, Tuple

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


def run_robustness_grid(
    df: pd.DataFrame,
    base_confounder_cols: List[str],
    country_dummy_cols: List[str],
    outcome_col: str = "total_gain_w",
    cv: int = 5,
    random_state: int = 42,
) -> pd.DataFrame:
    """
    Run the full robustness specification grid.

    Grid dimensions:
    1. Treatment: {total_self_trxn (count), self_trade_ratio}
    2. Nuisance learner: {histgbr, lightgbm, lasso}
    3. Sample: {full, tenure >= 23 weeks}
    4. Country controls: {with, without}

    Parameters
    ----------
    df : pd.DataFrame
        Analysis-ready DataFrame.
    base_confounder_cols : list of str
        Non-country confounder columns.
    country_dummy_cols : list of str
        Country dummy columns.
    outcome_col : str
        Outcome column.
    cv : int
        Cross-fitting folds.
    random_state : int
        Random seed.

    Returns
    -------
    pd.DataFrame
        Results table with one row per specification.
    """
    from src.models import fit_linear_dml

    treatments = [
        ("total_self_trxn", "count"),
        ("self_trade_ratio", "ratio"),
    ]
    nuisance_learners = ["histgbr", "lightgbm", "lasso"]
    samples = [("full", None), ("tenure_ge_23", 23)]
    country_options = [("with_country", True), ("without_country", False)]

    results = []
    spec_id = 0

    total_specs = (
        len(treatments) * len(nuisance_learners)
        * len(samples) * len(country_options)
    )
    logger.info(f"Running robustness grid: {total_specs} specifications")

    for (t_col, t_label), learner, (s_label, tenure_min), (c_label, use_country) in (
        itertools.product(treatments, nuisance_learners, samples, country_options)
    ):
        spec_id += 1
        logger.info(
            f"  Spec {spec_id}/{total_specs}: T={t_label}, nuisance={learner}, "
            f"sample={s_label}, country={c_label}"
        )

        try:
            df_spec = df.copy()

            # Filter sample
            if tenure_min is not None:
                df_spec = df_spec[df_spec["tenure_weeks"] >= tenure_min]

            if len(df_spec) < 1000:
                logger.warning(
                    f"  Skipping spec {spec_id}: only {len(df_spec)} rows"
                )
                continue

            # Build confounder matrix
            conf_cols = list(base_confounder_cols)
            if use_country:
                conf_cols = conf_cols + [
                    c for c in country_dummy_cols if c in df_spec.columns
                ]

            # For linear nuisance learners, apply log1p/asinh transforms
            if learner in ("lasso", "ridge"):
                df_spec = _apply_linear_transforms(df_spec, conf_cols)
                conf_cols = _get_transformed_col_names(conf_cols, df_spec.columns)

            # Check for valid data — explicit narrow column subset
            valid_mask = df_spec[t_col].notna() & df_spec[outcome_col].notna()
            for c in conf_cols:
                if c in df_spec.columns:
                    valid_mask = valid_mask & df_spec[c].notna()
            df_spec = df_spec[valid_mask]

            if len(df_spec) < 1000:
                logger.warning(
                    f"  Skipping spec {spec_id}: only {len(df_spec)} valid rows"
                )
                continue

            Y = df_spec[outcome_col].values.astype(np.float64)
            T = df_spec[t_col].values.astype(np.float64)
            existing_cols = [c for c in conf_cols if c in df_spec.columns]
            W = df_spec[existing_cols].values.astype(np.float64)

            _, res = fit_linear_dml(
                Y=Y, T=T, W=W,
                nuisance_learner=learner,
                cv=cv,
                random_state=random_state,
            )

            results.append({
                "spec_id": spec_id,
                "treatment_type": t_label,
                "nuisance_learner": learner,
                "sample": s_label,
                "country_controls": c_label,
                "ate": res["ate"],
                "se": res["ate_se"],
                "ci_lower": res["ate_ci_lower"],
                "ci_upper": res["ate_ci_upper"],
                "pvalue": res["ate_pvalue"],
                "n": res["n"],
            })

        except Exception as e:
            logger.error(f"  Spec {spec_id} failed: {e}")
            results.append({
                "spec_id": spec_id,
                "treatment_type": t_label,
                "nuisance_learner": learner,
                "sample": s_label,
                "country_controls": c_label,
                "ate": np.nan,
                "se": np.nan,
                "ci_lower": np.nan,
                "ci_upper": np.nan,
                "pvalue": np.nan,
                "n": 0,
                "error": str(e),
            })

    results_df = pd.DataFrame(results)
    logger.info(f"Robustness grid complete: {len(results_df)} specifications")
    return results_df


def _apply_linear_transforms(
    df: pd.DataFrame, conf_cols: List[str]
) -> pd.DataFrame:
    """
    Apply log1p/asinh transforms for linear nuisance learners.

    Rules from the decision table:
    - Non-negative count variables: log1p()
    - Variables that can be negative (gain_difference_l1): asinh()
    - Binary variables (greater_than_ten_l1): no transform
    - tenure_weeks: log1p() if skewed
    """
    df = df.copy()

    negative_vars = {"gain_difference_l1", "gain_difference_l1_demeaned"}
    binary_vars = {"greater_than_ten_l1"}
    no_transform = {"period"}

    for col in conf_cols:
        if col not in df.columns:
            continue
        if col in binary_vars or col in no_transform or col.startswith("country_"):
            continue

        new_col = f"{col}_transformed"
        if col in negative_vars:
            df[new_col] = np.arcsinh(df[col])
        else:
            df[new_col] = np.log1p(df[col].clip(lower=0))

    return df


def _get_transformed_col_names(
    original_cols: List[str], available_cols: pd.Index
) -> List[str]:
    """Map column names to their transformed versions if available."""
    binary_vars = {"greater_than_ten_l1"}
    no_transform = {"period"}

    result = []
    for col in original_cols:
        transformed = f"{col}_transformed"
        if col in binary_vars or col in no_transform or col.startswith("country_"):
            if col in available_cols:
                result.append(col)
        elif transformed in available_cols:
            result.append(transformed)
        elif col in available_cols:
            result.append(col)
    return result


def overlap_check(
    T: np.ndarray,
    W: np.ndarray,
    feature_names: Optional[List[str]] = None,
    n_bins: int = 10,
) -> Dict[str, Any]:
    """
    Check overlap/positivity: treatment distribution across confounder quantiles.

    For each confounder, bins observations into quantiles and checks whether
    the treatment varies sufficiently across bins (required for identification).
    """
    T = np.asarray(T).ravel()
    W = np.asarray(W)

    if feature_names is None:
        feature_names = [f"W{i}" for i in range(W.shape[1])]

    diagnostics = {}
    for j, name in enumerate(feature_names):
        w_j = W[:, j]
        valid = np.isfinite(w_j) & np.isfinite(T)
        w_j, t_j = w_j[valid], T[valid]

        try:
            bins = pd.qcut(w_j, q=n_bins, duplicates="drop")
            bin_stats = (
                pd.DataFrame({"T": t_j, "bin": bins})
                .groupby("bin")["T"]
                .agg(["mean", "std", "count"])
            )
            diagnostics[name] = {
                "bin_means": bin_stats["mean"].tolist(),
                "bin_stds": bin_stats["std"].tolist(),
                "bin_counts": bin_stats["count"].tolist(),
                "t_range_across_bins": float(
                    bin_stats["mean"].max() - bin_stats["mean"].min()
                ),
            }
        except Exception as e:
            logger.warning(f"Overlap check failed for {name}: {e}")
            diagnostics[name] = {"error": str(e)}

    return diagnostics


def outlier_exclusion(
    df: pd.DataFrame,
    treatment_col: str = "total_self_trxn",
    outcome_col: str = "total_gain_w",
    sd_threshold: float = 3.0,
) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """
    Exclude observations where treatment or outcome exceeds sd_threshold SDs.

    Returns filtered DataFrame and exclusion stats.
    """
    stats = {"original_n": len(df)}

    for col in [treatment_col, outcome_col]:
        mean = df[col].mean()
        std = df[col].std()
        lower = mean - sd_threshold * std
        upper = mean + sd_threshold * std
        outlier_mask = (df[col] < lower) | (df[col] > upper)
        n_outliers = int(outlier_mask.sum())
        stats[f"{col}_outliers"] = n_outliers
        stats[f"{col}_outlier_pct"] = float(n_outliers / len(df))
        df = df[~outlier_mask].copy()

    stats["final_n"] = len(df)
    stats["total_removed"] = stats["original_n"] - stats["final_n"]
    logger.info(
        f"Outlier exclusion (>{sd_threshold} SD): "
        f"{stats['original_n']:,} → {stats['final_n']:,} "
        f"(removed {stats['total_removed']:,})"
    )

    return df, stats


def heckman_selection(
    df_full: pd.DataFrame,
    df_active: pd.DataFrame,
    selection_covariates: Optional[List[str]] = None,
) -> Dict[str, Any]:
    """
    Heckman two-step correction for the active-week selection restriction.

    Step 1: Probit model for P(active=1 | covariates) on the full panel
    Step 2: Compute inverse Mills ratio (IMR)

    Returns dict with IMR statistics and selection model results.
    """
    from scipy.stats import norm
    import statsmodels.api as sm

    results = {}

    df_full = df_full.copy()
    df_full["_active"] = (df_full["total_trxn"] > 0).astype(int)

    if selection_covariates is None:
        selection_covariates = ["tenure_weeks", "period"]

    # Filter to rows with valid covariates — explicit subset
    valid_mask = df_full[selection_covariates + ["_active"]].notna().all(axis=1)
    valid = df_full[valid_mask]

    if len(valid) < 100:
        logger.warning("Too few valid rows for Heckman selection model")
        return {"error": "insufficient data"}

    X_sel = sm.add_constant(valid[selection_covariates].values)
    y_sel = valid["_active"].values

    try:
        probit = sm.Probit(y_sel, X_sel).fit(disp=0)
        results["probit_converged"] = True
        results["probit_pseudo_r2"] = float(probit.prsquared)

        # Compute IMR
        xb = probit.predict(X_sel)
        xb_clipped = np.clip(xb, 0.001, 0.999)
        imr = norm.pdf(norm.ppf(xb_clipped)) / xb_clipped
        imr = np.where(np.isfinite(imr), imr, 0)

        results["imr_mean"] = float(np.mean(imr[y_sel == 1]))
        results["imr_std"] = float(np.std(imr[y_sel == 1]))
        results["selection_rate"] = float(y_sel.mean())
        results["n_full"] = int(len(y_sel))
        results["n_active"] = int(y_sel.sum())

    except Exception as e:
        logger.error(f"Heckman probit failed: {e}")
        results["probit_converged"] = False
        results["error"] = str(e)

    return results


def dowhy_refutation(
    df: pd.DataFrame,
    treatment_col: str = "total_self_trxn",
    outcome_col: str = "total_gain_w",
    confounder_cols: Optional[List[str]] = None,
    max_rows: int = 50000,
) -> Dict[str, Any]:
    """
    DoWhy refutation tests for the causal estimate.

    Runs:
    1. Placebo treatment (random permutation of T)
    2. Random common cause (add noise variable)
    3. Data subset (resample)
    """
    try:
        from dowhy import CausalModel
    except ImportError:
        logger.warning("DoWhy not available — skipping refutation tests")
        return {"error": "dowhy not installed"}

    results = {}

    # Subsample if needed
    if len(df) > max_rows:
        df = df.sample(n=max_rows, random_state=42)
        results["subsampled"] = True
        results["n"] = max_rows
    else:
        results["subsampled"] = False
        results["n"] = len(df)

    if confounder_cols is None:
        confounder_cols = []

    try:
        model = CausalModel(
            data=df,
            treatment=treatment_col,
            outcome=outcome_col,
            common_causes=confounder_cols,
        )

        identified = model.identify_effect(proceed_when_unidentifiable=True)
        estimate = model.estimate_effect(
            identified,
            method_name="backdoor.linear_regression",
        )
        results["baseline_estimate"] = float(estimate.value)

        # Refutation 1: Placebo treatment
        try:
            refute_placebo = model.refute_estimate(
                identified, estimate,
                method_name="placebo_treatment_refuter",
                placebo_type="permute",
                num_simulations=50,
            )
            results["placebo_effect"] = float(refute_placebo.new_effect)
            if hasattr(refute_placebo, "refutation_result") and refute_placebo.refutation_result:
                results["placebo_pvalue"] = float(
                    refute_placebo.refutation_result.get("p_value", np.nan)
                )
        except Exception as e:
            results["placebo_error"] = str(e)

        # Refutation 2: Random common cause
        try:
            refute_random = model.refute_estimate(
                identified, estimate,
                method_name="random_common_cause",
                num_simulations=50,
            )
            results["random_cause_effect"] = float(refute_random.new_effect)
        except Exception as e:
            results["random_cause_error"] = str(e)

        # Refutation 3: Data subset
        try:
            refute_subset = model.refute_estimate(
                identified, estimate,
                method_name="data_subset_refuter",
                subset_fraction=0.8,
                num_simulations=50,
            )
            results["subset_effect"] = float(refute_subset.new_effect)
        except Exception as e:
            results["subset_error"] = str(e)

    except Exception as e:
        logger.error(f"DoWhy refutation failed: {e}")
        results["error"] = str(e)

    return results
