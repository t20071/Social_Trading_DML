
"""
Sensitivity to unobserved confounding (Cinelli & Hazlett 2020, JRSS-B).

- robustness_value(): RV from a partial Cohen's f
- sensitivity_analysis(): RV for the DML final stage (with clustered SEs for
  the significance version) and benchmark bounds against observed controls
"""

import logging
from typing import Any, Dict

import numpy as np

logger = logging.getLogger(__name__)


def robustness_value(f: float) -> float:
    """Robustness value from a partial Cohen's f (Cinelli & Hazlett 2020, eq. 18)."""
    if f <= 0:
        return 0.0
    return 0.5 * (np.sqrt(f ** 4 + 4 * f ** 2) - f ** 2)


def sensitivity_analysis(
    arrays: Dict[str, Any],
    dml_result: Dict[str, Any],
    alpha: float = 0.05,
    kd: float = 1.0,
    ky: float = 1.0,
) -> Dict[str, Any]:
    """
    Cinelli & Hazlett (2020, JRSS-B) sensitivity to unobserved confounding.

    1. Robustness values (RV) for the DML final stage: the minimum share of
       residual variance of BOTH treatment and outcome an unobserved confounder
       would need to explain to bring the estimate to zero (RV_q=1) or to make
       it insignificant (RV_q=1,alpha).
    2. Benchmarks on the linear two-way FE regression (same demeaned data):
       for each observed control X_j, the bias a confounder `kd`/`ky` times as
       strong as X_j would induce, and the adjusted estimate.

    The partial R^2 quantities use classical OLS t-statistics, as in the
    original method. RV_q=1 does not depend on the SE. For RV_q=1,alpha the
    significance threshold is scaled by the ratio of the clustered to the
    classical SE (trader-clustered, and two-way if available), since the
    classical SE understates uncertainty here.
    """
    import statsmodels.api as sm
    from scipy import stats

    Y, T, W = arrays["Y"], arrays["T"], arrays["W"]
    names = arrays["W_names"]

    # (1) RV for the DML final stage
    t = dml_result["t_stat_classical"]
    dof = dml_result["dof"]
    f = abs(t) / np.sqrt(dof)
    t_crit = stats.t.ppf(1 - alpha / 2, dof - 1)
    out = {
        "dml_estimate": dml_result["ate"],
        "dml_partial_r2_ty": dml_result["partial_r2_ty"],
        "rv_q1": float(robustness_value(f)),
        "rv_q1_alpha_classical_se": float(robustness_value(f - t_crit / np.sqrt(dof - 1))),
    }
    se_classical = abs(dml_result["ate"] / t) if t != 0 else np.nan
    for key, se_key in [("rv_q1_alpha", "ate_se"), ("rv_q1_alpha_twoway", "ate_se_twoway")]:
        if dml_result.get(se_key):
            ratio = dml_result[se_key] / se_classical
            out[key] = float(robustness_value(f - ratio * t_crit / np.sqrt(dof - 1)))
            out[f"{key}_se_ratio"] = float(ratio)

    # (2) Benchmarks from the linear FE regression
    XY = sm.add_constant(np.column_stack([T, W]), has_constant="add")
    fit_y = sm.OLS(Y, XY).fit()
    fit_d = sm.OLS(T, sm.add_constant(W, has_constant="add")).fit()
    dof_y = fit_y.df_resid
    est, se = float(fit_y.params[1]), float(fit_y.bse[1])
    out.update({
        "ols_fe_estimate": est,
        "ols_fe_se_classical": se,
        "ols_fe_dof": float(dof_y),
    })

    benchmarks = []
    for j, name in enumerate(names):
        t_dj = float(fit_d.tvalues[j + 1])
        t_yj = float(fit_y.tvalues[j + 2])
        r2dxj = t_dj ** 2 / (t_dj ** 2 + fit_d.df_resid)
        r2yxj = t_yj ** 2 / (t_yj ** 2 + dof_y)

        r2dz = kd * r2dxj / (1 - r2dxj)
        if r2dz >= 1:
            benchmarks.append({"control": name, "error": "implausible bound (R2dz>=1)"})
            continue
        r2zxj = kd * r2dxj ** 2 / ((1 - kd * r2dxj) * (1 - r2dxj))
        r2yz = ((np.sqrt(ky) + np.sqrt(r2zxj)) / np.sqrt(1 - r2zxj)) ** 2 \
            * (r2yxj / (1 - r2yxj))
        r2yz = min(r2yz, 1.0)
        bias = se * np.sqrt(r2yz * r2dz / (1 - r2dz)) * np.sqrt(dof_y)
        adjusted = np.sign(est) * (abs(est) - bias)
        benchmarks.append({
            "control": name,
            "partial_r2_with_T": float(r2dxj),
            "partial_r2_with_Y": float(r2yxj),
            "bound_r2_dz": float(r2dz),
            "bound_r2_yz": float(r2yz),
            "bias": float(bias),
            "adjusted_estimate": float(adjusted),
            "sign_flips": bool(np.sign(adjusted) != np.sign(est)),
        })
    out["benchmark_k"] = {"kd": kd, "ky": ky}
    out["benchmarks"] = benchmarks
    return out
