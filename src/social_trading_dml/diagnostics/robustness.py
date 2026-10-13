
"""
Robustness of the primary estimate (LinearDML on two-way FE data).

- run_robustness_grid(): 24 specifications
  {count, log1p count} x {HistGBR, LightGBM, Lasso}
  x {all traders, present >= 23 of 46 weeks} x {winsorized, asinh outcome}
- additional_checks(): re-estimates that each change one design choice
  (FE structure, outcome definition and components, crowding out, trimming,
  ratio treatment, first differences, exact FD and correlated random effects,
  lagged outcome), with same-sample primary rows where the sample changes
- placebo_within_trader(): treatment permuted within trader (expected ~0)
- trim_extreme_treatment(): treatment-only trim used by additional_checks()
"""

import itertools
import logging
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from social_trading_dml.data.features import MIN_WEEKS_PRESENT
from social_trading_dml.diagnostics.common import fit_spec, sample_traders, summarize_result
from social_trading_dml.estimation.panel import build_fe_arrays, first_difference
from social_trading_dml.diagnostics.identification import week_block_label

logger = logging.getLogger(__name__)


def run_robustness_grid(
    df: pd.DataFrame,
    control_cols: List[str],
    cv: int = 5,
    random_state: int = 42,
    max_traders: Optional[int] = None,
) -> pd.DataFrame:
    """
    Run the 24-spec robustness grid on the primary two-way FE model.

    Grid dimensions:
    1. Treatment: total_self_trxn (count) | self_trade_count_log (log1p)
    2. Nuisance learner: histgbr | lightgbm | lasso
    3. Sample: all traders | traders present >= 23 of 46 weeks (Mani et al.)
    4. Outcome: total_gain_w (winsorized) | total_gain_asinh

    Country dummies are time-invariant and absorbed by trader FE, so they
    are not a grid dimension.

    Returns
    -------
    pd.DataFrame
        One row per specification (failed specs are dropped and logged).
    """
    df = sample_traders(df, max_traders, random_state)

    treatments = [("total_self_trxn", "count"), ("self_trade_count_log", "log1p")]
    learners = ["histgbr", "lightgbm", "lasso"]
    samples = [("all", None), (f"present_ge_{MIN_WEEKS_PRESENT}", MIN_WEEKS_PRESENT)]
    outcomes = [("total_gain_w", "winsorized"), ("total_gain_asinh", "asinh")]

    total = len(treatments) * len(learners) * len(samples) * len(outcomes)
    logger.info(f"Running robustness grid: {total} specifications")

    results = []
    spec_id = 0
    for (s_label, min_weeks) in samples:
        df_s = df if min_weeks is None else df[df["weeks_present"] >= min_weeks]
        for (t_col, t_label), (y_col, y_label) in itertools.product(treatments, outcomes):
            arrays = build_fe_arrays(df_s, t_col, y_col, control_cols)
            for learner in learners:
                spec_id += 1
                logger.info(
                    f"  Spec {spec_id}/{total}: T={t_label}, Y={y_label}, "
                    f"nuisance={learner}, sample={s_label}"
                )
                try:
                    res = fit_spec(arrays, learner, cv, random_state,
                                    cache_name="grid")
                except Exception as e:
                    logger.error(f"  Spec {spec_id} failed: {e}")
                    continue
                results.append({
                    "spec_id": spec_id,
                    "treatment_type": t_label,
                    "outcome": y_label,
                    "nuisance_learner": learner,
                    "sample": s_label,
                    **summarize_result(res),
                })

    results_df = pd.DataFrame(results)
    logger.info(f"Robustness grid complete: {len(results_df)}/{total} specifications")
    return results_df


def additional_checks(
    df: pd.DataFrame,
    control_cols: List[str],
    country_dummy_cols: List[str],
    primary: Dict[str, Any],
    cv: int = 5,
    random_state: int = 42,
    max_traders: Optional[int] = None,
) -> pd.DataFrame:
    """
    Re-estimate the primary model changing one design choice at a time.

    Every row is a fitted HistGBR LinearDML with trader-grouped cross-fitting
    and trader-clustered SEs. `primary` is the primary-spec result dict and is
    included as the first row for reference.
    """
    df = sample_traders(df, max_traders, random_state)
    T, Y = "total_self_trxn", "total_gain_w"
    rows = [{"check": "Primary: two-way FE, winsorized outcome", **summarize_result(primary)}]

    def run(label, data, t_col, y_col, controls, fe="twoway", note=None):
        logger.info(f"  Check: {label}")
        try:
            arrays = build_fe_arrays(data, t_col, y_col, controls, fe=fe)
            res = fit_spec(arrays, "histgbr", cv, random_state, cache_name="check")
            rows.append({"check": label, **summarize_result(res), "note": note})
        except Exception as e:
            logger.error(f"  Check '{label}' failed: {e}")
            rows.append({"check": label, "error": str(e)})

    # Fixed-effects structure
    run("No FE (pooled levels + country dummies)", df, T, Y,
        control_cols + ["tenure_weeks"] + country_dummy_cols, fe="none")
    run("Trader FE only (no week FE)", df, T, Y, control_cols, fe="trader")

    # Outcome definition and decomposition (total = self + copied gain).
    # The raw components add up to the raw total; winsorized ones need not.
    run("Raw outcome (total_gain, not winsorized)", df, T, "total_gain", control_cols)
    for comp, label in [("total_self_gain", "self-trade"), ("total_copied_gain", "copied-trade")]:
        if comp in df.columns:
            run(f"Outcome: {label} gain only (raw)", df, T, comp, control_cols)
        if f"{comp}_w" in df.columns:
            run(f"Outcome: {label} gain only (winsorized)", df, T, f"{comp}_w", control_cols)

    # Reallocation (crowding out): does self-trading displace copy trading?
    # A null effect on copied *gain* is uninformative when copied trades earn
    # about zero, so test the copied-trade *count* directly.
    if "total_copied_trxn" in df.columns:
        run("Outcome: copied-trade count (crowding out)", df, T, "total_copied_trxn",
            control_cols)

    # Extreme treatment weeks trimmed (treatment only; never trim on outcome)
    df_trim, trim_stats = trim_extreme_treatment(df, treatment_col=T)
    run(f"Self-trades > p99.9 of positive weeks ({trim_stats['cutoff']:.0f}) "
        f"excluded ({trim_stats['n_removed']:,} rows)", df_trim, T, Y, control_cols,
        note=f"removes {trim_stats['share_of_treatment_removed']:.1%} of all self-trades")

    # Ratio treatment on active weeks only (selected sample)
    active = df[df["total_trxn"] > 0]
    run("Ratio treatment (self/total trades), active weeks only", active,
        "self_trade_ratio", Y, control_cols)

    # First-difference DML (Clarke & Polselli 2025): differences out trader FE;
    # week FE handled by demeaning the differenced data by week. Differencing
    # drops each trader's first analysis week, so the primary model is refit on
    # the same rows for comparison.
    fd_cols = [T, Y] + control_cols
    df_fd = first_difference(df, fd_cols)
    fd_names = [f"{c}_fd" for c in fd_cols]
    df_fd = df_fd.dropna(subset=fd_names)
    df_fd_w = df_fd.copy()
    for c in fd_names:
        df_fd_w[c] = df_fd[c] - df_fd.groupby("period")[c].transform("mean")
    run("Primary on first-difference sample (same rows)", df.loc[df_fd.index], T, Y,
        control_cols)
    run("First-difference DML, nuisances on differenced controls", df_fd_w,
        f"{T}_fd", f"{Y}_fd", [f"{c}_fd" for c in control_cols], fe="none",
        note="approximation: nuisances see only the differenced controls")

    # Exact FD-DML (Clarke & Polselli): nuisances learn from the control levels
    # in t and t-1, not only their difference
    df_x = df_fd.copy()
    g = df.sort_values(["trader_id", "period"]).groupby("trader_id")
    for c in control_cols:
        df_x[f"{c}_prev"] = g[c].shift(1).reindex(df_x.index)
    prev_cols = [f"{c}_prev" for c in control_cols]
    df_x = df_x.dropna(subset=prev_cols)
    run("First-difference DML, nuisances on controls in t and t-1 (exact FD)", df_x,
        f"{T}_fd", f"{Y}_fd", control_cols + prev_cols, fe="week",
        note="Clarke & Polselli exact first-difference form")

    # Correlated random effects (Mundlak): week effects removed, trader means
    # of the controls and the treatment added as nuisance features instead of
    # within-demeaning
    df_cre = df.copy()
    mean_cols = []
    for c in control_cols + [T]:
        df_cre[f"{c}_tmean"] = df_cre.groupby("trader_id")[c].transform("mean")
        mean_cols.append(f"{c}_tmean")
    run("Correlated random effects (Mundlak) DML", df_cre, T, Y,
        control_cols + mean_cols, fe="week",
        note="trader means of controls and treatment as nuisance features")

    # Dynamic: add lagged outcome (Nickell-biased with T=45; comparison only).
    # Use the lag computed on the full panel before sample filters if present.
    df_dyn = df.sort_values(["trader_id", "period"]).copy()
    if "total_gain_w_l1" not in df_dyn.columns:
        df_dyn["total_gain_w_l1"] = df_dyn.groupby("trader_id")[Y].shift(1)
    df_dyn = df_dyn.dropna(subset=["total_gain_w_l1"])
    if len(df_dyn) < len(df):
        run("Primary on dynamic-check sample (same rows)", df_dyn, T, Y, control_cols)
    run("Dynamic: + lagged outcome (Nickell-biased, comparison only)", df_dyn, T, Y,
        control_cols + ["total_gain_w_l1"])

    return pd.DataFrame(rows)


def placebo_within_trader(
    df: pd.DataFrame,
    control_cols: List[str],
    n_reps: int = 3,
    cv: int = 5,
    random_state: int = 42,
    max_traders: Optional[int] = None,
) -> pd.DataFrame:
    """
    Placebo: permute the treatment within each trader x period block, then
    refit the primary model. This keeps each trader's treatment distribution
    and its concentration in calendar blocks, but breaks the week-to-week link
    with gains; estimates should be centred on zero. It checks the pipeline,
    not identification.
    """
    df = sample_traders(df, max_traders, random_state)
    rng = np.random.RandomState(random_state)
    rows = []
    for rep in range(n_reps):
        d = df[["trader_id", "period", "total_self_trxn", "total_gain_w"]
               + control_cols].copy()
        block = week_block_label(d["period"].to_numpy())
        d["placebo_T"] = d.groupby([d["trader_id"], block])["total_self_trxn"].transform(
            lambda s: rng.permutation(s.to_numpy())
        )
        arrays = build_fe_arrays(d, "placebo_T", "total_gain_w", control_cols)
        res = fit_spec(arrays, "histgbr", cv, random_state + rep + 1)
        rows.append({"rep": rep + 1, **summarize_result(res)})
        logger.info(f"  Placebo rep {rep + 1}: ATE={res['ate']:.4f} (SE {res['ate_se']:.4f})")
    return pd.DataFrame(rows)


def trim_extreme_treatment(
    df: pd.DataFrame,
    treatment_col: str = "total_self_trxn",
    quantile: float = 0.999,
) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """
    Drop weeks whose treatment exceeds the `quantile` of POSITIVE treatment
    values. Trims on the treatment only: trimming on the outcome selects on
    the dependent variable and biases the slope toward zero. (A 3-SD rule on
    a count that is zero in 94% of weeks would drop most self-trading.)
    """
    pos = df.loc[df[treatment_col] > 0, treatment_col]
    cutoff = float(pos.quantile(quantile))
    mask = df[treatment_col] > cutoff
    stats = {
        "cutoff": cutoff,
        "n_removed": int(mask.sum()),
        "share_of_treatment_removed": float(
            df.loc[mask, treatment_col].sum() / df[treatment_col].sum()
        ),
    }
    logger.info(
        f"Treatment trim (> p{quantile * 100:g} of positive = {cutoff:.0f}): "
        f"removed {stats['n_removed']:,} rows, "
        f"{stats['share_of_treatment_removed']:.1%} of all self-trades"
    )
    return df[~mask], stats
