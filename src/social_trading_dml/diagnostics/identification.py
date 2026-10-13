
"""
Identification diagnostics.

- lead_test(): strict exogeneity (does next week's self-trading predict this
  week's gain?)
- week_decomposition(): the DML estimate as a weighted average of
  week-specific slopes, with leave-one-week-out estimates
- week_block_estimates() / week_block_label(): refits within calendar blocks
- overlap_check(): treatment distribution within confounder bins
"""

import logging
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from social_trading_dml.config import WEEK_BLOCKS
from social_trading_dml.diagnostics.common import fit_spec, summarize_result
from social_trading_dml.estimation.dml import week_inference
from social_trading_dml.estimation.inference import partial_out
from social_trading_dml.estimation.panel import build_fe_arrays

logger = logging.getLogger(__name__)


def lead_test(
    df: pd.DataFrame,
    control_cols: List[str],
    lead_col: str = "total_self_trxn_f1",
    cv: int = 5,
    random_state: int = 42,
) -> Dict[str, Any]:
    """
    Strict-exogeneity diagnostic: does NEXT week's self-trading predict this
    week's gain, conditional on this week's self-trading and the controls?

    Under strict exogeneity (needed for FE and FD estimators) the lead should
    have no effect. Reports the linear two-way FE coefficient on the lead
    (trader-clustered), and the DML coefficient on T with and without the
    lead as an extra control on the same rows (rows without a lead, i.e. the
    last week, are dropped). `lead_col` should be computed on the full panel.
    """
    import statsmodels.api as sm

    T, Y = "total_self_trxn", "total_gain_w"
    d = df.dropna(subset=[lead_col])
    arrays = build_fe_arrays(d, T, Y, control_cols + [lead_col])
    j = arrays["W_names"].index(lead_col)
    X = np.column_stack([arrays["T"], arrays["W"]])
    fit = sm.OLS(arrays["Y"], X).fit(
        cov_type="cluster", cov_kwds={"groups": pd.factorize(arrays["groups"])[0]}
    )
    # Same coefficient by Frisch-Waugh-Lovell, for week-level inference
    others = np.delete(X, j + 1, axis=1)
    lead_res = partial_out(arrays["W"][:, j], others)
    y_res = partial_out(arrays["Y"], others)
    week = week_inference(y_res, lead_res, arrays["groups"], arrays["periods"])
    out = {
        "n": int(len(arrays["Y"])),
        "periods_used": [int(d["period"].min()), int(d["period"].max())],
        "linear_fe_lead_coef": float(fit.params[j + 1]),
        "linear_fe_lead_se": float(fit.bse[j + 1]),
        "linear_fe_lead_t": float(fit.tvalues[j + 1]),
        "linear_fe_lead_pvalue": float(fit.pvalues[j + 1]),
        "linear_fe_lead_se_week": week["ate_se_week"],
        "linear_fe_lead_t_week": (float(fit.params[j + 1]) / week["ate_se_week"]
                                  if week["ate_se_week"] else None),
        "linear_fe_lead_pvalue_week_wild": week["ate_pvalue_week_wild"],
        "linear_fe_lead_se_twoway": week["ate_se_twoway"],
        "linear_fe_lead_pvalue_twoway": week["ate_pvalue_twoway"],
        "effective_time_clusters": week["effective_time_clusters"],
        "linear_fe_T_coef": float(fit.params[0]),
    }
    base = build_fe_arrays(d, T, Y, control_cols)
    res0 = fit_spec(base, "histgbr", cv, random_state, cache_name="check")
    res1 = fit_spec(arrays, "histgbr", cv, random_state, cache_name="check")
    out.update({
        "dml_T_same_sample": res0["ate"],
        "dml_T_same_sample_se": res0["ate_se"],
        "dml_T_with_lead_control": res1["ate"],
        "dml_T_with_lead_control_se": res1["ate_se"],
    })
    week_txt = (f", t={out['linear_fe_lead_t_week']:.1f} week-clustered, "
                f"wild p={out['linear_fe_lead_pvalue_week_wild']:.3g}"
                if out["linear_fe_lead_t_week"] is not None else "")
    logger.info(
        f"Lead test: lead coef {out['linear_fe_lead_coef']:.3f} "
        f"(t={out['linear_fe_lead_t']:.1f} trader-clustered{week_txt}); "
        f"DML T {res0['ate']:.3f} -> {res1['ate']:.3f} with lead control (diagnostic only)"
    )
    return out


def week_decomposition(
    Y_res: np.ndarray,
    T_res: np.ndarray,
    periods: np.ndarray,
) -> pd.DataFrame:
    """
    Decompose the DML final-stage estimate into week-specific slopes.

    theta = sum_w weight_w * slope_w, with weight_w = sum_{i in w} T_res^2 /
    sum T_res^2 and slope_w = sum_{i in w} T_res*Y_res / sum_{i in w} T_res^2.

    Also gives, per week, a heteroskedasticity-robust SE and 95% CI for the
    week-specific slope (each trader appears once per week, so this equals a
    trader-clustered SE within the week), and the leave-one-week-out estimate
    (final stage only, nuisances held fixed).
    """
    Y_res = np.asarray(Y_res, dtype=np.float64).ravel()
    T_res = np.asarray(T_res, dtype=np.float64).ravel()
    d = pd.DataFrame({"period": periods, "t": T_res, "y": Y_res,
                      "tt": T_res ** 2, "ty": T_res * Y_res})
    g = d.groupby("period")[["tt", "ty"]].sum()
    tot_tt, tot_ty = g["tt"].sum(), g["ty"].sum()
    slope = g["ty"] / g["tt"].replace(0, np.nan)

    # HC0 sandwich per week: sum(t^2 e^2) / (sum t^2)^2, e = y - slope_w * t
    e = d["y"] - d["period"].map(slope) * d["t"]
    meat = (d["tt"] * e ** 2).groupby(d["period"]).sum()
    se = np.sqrt(meat) / g["tt"].replace(0, np.nan)

    return pd.DataFrame({
        "period": g.index.astype(int),
        "weight": (g["tt"] / tot_tt).to_numpy(),
        "slope": slope.to_numpy(),
        "slope_se": se.to_numpy(),
        "slope_ci_lower": (slope - 1.96 * se).to_numpy(),
        "slope_ci_upper": (slope + 1.96 * se).to_numpy(),
        "estimate_leave_week_out": ((tot_ty - g["ty"]) / (tot_tt - g["tt"])).to_numpy(),
        "n": d.groupby("period").size().to_numpy(),
    })


def week_block_estimates(
    df: pd.DataFrame,
    control_cols: List[str],
    blocks: Dict[str, Tuple[int, int]],
    cv: int = 5,
    random_state: int = 42,
) -> pd.DataFrame:
    """
    Refit the primary model separately within calendar-week blocks
    {label: (first_period, last_period)} (FE, nuisances and SEs all
    re-estimated within the block).
    """
    rows = []
    for label, (lo, hi) in blocks.items():
        d = df[(df["period"] >= lo) & (df["period"] <= hi)]
        logger.info(f"  Week block {label} ({lo}-{hi}): {len(d):,} rows")
        arrays = build_fe_arrays(d, "total_self_trxn", "total_gain_w", control_cols)
        res = fit_spec(arrays, "histgbr", cv, random_state, cache_name="block")
        rows.append({"block": label, "first_period": lo, "last_period": hi,
                     "self_trades": int(d["total_self_trxn"].sum()), **summarize_result(res)})
    return pd.DataFrame(rows)



def week_block_label(periods: np.ndarray,
                     blocks: Dict[str, Tuple[int, int]] = WEEK_BLOCKS) -> np.ndarray:
    """Label each period with its calendar-week block name ('' if none)."""
    labels = np.full(len(periods), "", dtype=object)
    for name, (lo, hi) in blocks.items():
        labels[(periods >= lo) & (periods <= hi)] = name
    return labels


def overlap_check(
    T: np.ndarray,
    W: np.ndarray,
    feature_names: Optional[List[str]] = None,
    n_bins: int = 10,
) -> pd.DataFrame:
    """
    Overlap/positivity: treatment distribution within confounder bins.

    Bins are 'zero' plus deciles of the non-zero values (so zero-inflated
    counts do not collapse into one bin). For each bin, reports mean, SD,
    share of zero treatment and 99th percentile of T. Identification needs
    treatment variation (SD > 0, zero-share < 1) within every bin.
    """
    T = np.asarray(T).ravel()
    W = np.asarray(W)

    if feature_names is None:
        feature_names = [f"W{i}" for i in range(W.shape[1])]

    rows = []
    for j, name in enumerate(feature_names):
        w_j = W[:, j]
        valid = np.isfinite(w_j) & np.isfinite(T)
        w_v = w_j[valid]
        bins = np.full(len(w_v), "zero", dtype=object)
        nz = w_v != 0
        nz_values = np.unique(w_v[nz])
        if len(nz_values) <= n_bins:
            # binary / low-cardinality control: one bin per value
            bins[nz] = [f"value {v:g}" for v in w_v[nz]]
        else:
            q = pd.qcut(w_v[nz], q=n_bins, duplicates="drop", labels=False)
            bins[nz] = [f"q{int(b) + 1:02d}" for b in q]
        g = pd.DataFrame({"T": T[valid], "bin": bins}).groupby("bin")["T"]
        stats_df = pd.DataFrame({
            "mean_T": g.mean(),
            "sd_T": g.std(),
            "share_T_zero": g.apply(lambda s: float((s == 0).mean())),
            "p99_T": g.quantile(0.99),
            "n": g.size(),
        }).reset_index()
        stats_df.insert(0, "confounder", name)
        rows.append(stats_df)

    out = pd.concat(rows, ignore_index=True) if rows else pd.DataFrame()
    if len(out):
        bad = out[(out["sd_T"] <= 0) | (out["share_T_zero"] >= 1)]
        if len(bad):
            logger.warning(f"Overlap: {len(bad)} confounder bins with no treatment variation")
    return out


def regime_break(
    week_dec: pd.DataFrame,
    min_segment: int = 5,
) -> Dict[str, Any]:
    """
    Best single break in the week-specific contributions (exploratory).

    For each candidate split, fits a separate inverse-variance-weighted mean
    to the weeks before and after it and keeps the split with the smallest
    weighted sum of squared residuals. Data-driven, so treat the location as
    descriptive, not as a test.
    """
    d = week_dec.dropna(subset=["slope", "slope_se"]).sort_values("period")
    d = d[d["slope_se"] > 0]
    w = 1.0 / d["slope_se"] ** 2
    best = None
    periods = d["period"].to_numpy()
    for k in periods[min_segment - 1:-min_segment]:
        a, b = d["period"] <= k, d["period"] > k
        ma = np.average(d.loc[a, "slope"], weights=w[a])
        mb = np.average(d.loc[b, "slope"], weights=w[b])
        ssr = float((w[a] * (d.loc[a, "slope"] - ma) ** 2).sum()
                    + (w[b] * (d.loc[b, "slope"] - mb) ** 2).sum())
        if best is None or ssr < best["ssr"]:
            best = {"break_after_period": int(k), "mean_before": float(ma),
                    "mean_after": float(mb), "ssr": ssr}
    if best is not None:
        best["method"] = "single break, inverse-variance-weighted means, min segment %d" % min_segment
    return best or {}


def week_aggregates(week_dec: pd.DataFrame, self_trades: pd.Series) -> Dict[str, Any]:
    """
    Three ways to average the week contributions: the DML weighting (= the
    pooled estimate), equal weight per week, and weight by self-trades.
    """
    d = week_dec.dropna(subset=["slope"]).set_index("period")
    trades = self_trades.reindex(d.index).fillna(0)
    return {
        "variance_weighted": float((d["weight"] * d["slope"]).sum() / d["weight"].sum()),
        "equal_week": float(d["slope"].mean()),
        "trade_weighted": float((trades * d["slope"]).sum() / trades.sum()),
        "n_weeks": int(len(d)),
    }

