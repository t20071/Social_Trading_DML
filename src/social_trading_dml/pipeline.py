"""
End-to-end analysis pipeline: self-trading intensity → weekly trader gains.

Stages
------
1. load_panel()              cleaned trader-week panel (parquet cache or raw CSV)
2. prepare_sample()          features, treatment/outcome, analysis sample,
                             period calendar and activity
3. estimate_main_models()    naive OLS; LinearDML without FE, with trader FE and
                             with trader + week FE (primary); linear FE OLS;
                             trader-level cross-section; week contributions,
                             aggregates, regime break and period-block refits
4. estimate_heterogeneity()  CausalForestDML on one half of traders; GATES,
                             BLP calibration, moderator and block tests on the
                             held-out half
5. run_robustness()          sensitivity, lead test, Anderson-Hsiao (feedback-
                             robust), overlap, 24-spec grid, single-change
                             checks, within-trader placebo
6. make_figures()            figures (default: reports/figures/)

Primary specification: LinearDML on two-way (trader + week) demeaned data,
cross-fitting folds that never split a trader. Inference is reported three
ways: trader-clustered, week wild cluster bootstrap and trader x week
two-way, because the identifying variation is concentrated in a few weeks.

Usage
-----
    python run_pipeline.py [options]          # or: python -m social_trading_dml
    python run_pipeline.py --help
"""

import argparse
import json
import logging
import subprocess
import sys
import time
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from social_trading_dml import __version__, config
from social_trading_dml.data.features import (
    CORE_LAG_COLUMNS, build_features, get_country_dummies,
)
from social_trading_dml.data.loading import (
    clean_data, get_missingness_report, load_processed, load_raw_data,
    print_cleaning_summary, save_processed, verify_column_diagnostics, verify_counts,
)
from social_trading_dml.data.variables import (
    activity_descriptives, collapse_to_trader_level, compute_outcome, compute_treatment,
    get_analysis_sample, outcome_descriptives, period_activity,
)
from social_trading_dml.diagnostics.identification import (
    lead_test, overlap_check, regime_break, week_aggregates, week_block_estimates,
    week_block_label, week_decomposition,
)
from social_trading_dml.diagnostics.robustness import (
    additional_checks, placebo_within_trader, run_robustness_grid,
)
from social_trading_dml.diagnostics.sensitivity import sensitivity_analysis
from social_trading_dml.estimation import cache
from social_trading_dml.estimation.dml import (
    fit_linear_dml, linear_fe_ols, naive_ols, prepare_model_arrays,
)
from social_trading_dml.estimation.dynamic import anderson_hsiao
from social_trading_dml.estimation.heterogeneity import (
    blp_calibration, blp_moderators, fit_causal_forest, gates_test, group_effects,
)
from social_trading_dml.estimation.panel import build_fe_arrays
from social_trading_dml.io import save_csv, save_json, update_manifest
from social_trading_dml.viz import figures

TREATMENT = config.TREATMENT
OUTCOME = config.OUTCOME
HETEROGENEITY_COLS = config.HETEROGENEITY_COLS
WEEK_BLOCKS = config.WEEK_BLOCKS

logger = logging.getLogger("pipeline")


# ════════════════════════════════════════════════════════════════════════════
# CLI and output bookkeeping
# ════════════════════════════════════════════════════════════════════════════

def parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Causal ML pipeline: effect of self-trading intensity on trader gains.",
    )
    g = parser.add_argument_group("data")
    g.add_argument("--data-path", type=Path, default=config.default_raw_data_path(),
                   help="Raw CSV (default: data/raw/trader_data_exl.csv, else repo root).")
    g.add_argument("--reload-data", action="store_true",
                   help="Re-read and re-clean the raw CSV instead of the cached parquet.")
    g.add_argument("--subsample-traders", type=int, default=None,
                   help="Keep N randomly chosen traders (whole histories) for fast iteration.")

    g = parser.add_argument_group("estimation")
    g.add_argument("--refit-models", action="store_true",
                   help="Refit models even if cached fits exist.")
    g.add_argument("--force-refit", action="store_true",
                   help="Shorthand for --refit-models --reload-data.")
    g.add_argument("--cache-dir", type=Path, default=config.CACHE_DIR,
                   help="Model cache directory (default: data/cache/).")
    g.add_argument("--skip-causal-forest", action="store_true",
                   help="Skip CausalForestDML and the held-out heterogeneity tests.")
    g.add_argument("--max-cf-samples", type=int, default=200_000,
                   help="Max rows (whole traders) used to fit the forest (default: 200000).")

    g = parser.add_argument_group("robustness")
    g.add_argument("--skip-robustness", action="store_true",
                   help="Skip sensitivity, lead test, Anderson-Hsiao, overlap, grid, "
                        "checks and placebo.")
    g.add_argument("--robustness-max-traders", type=int, default=None,
                   help="Run the grid, checks and placebo on N random traders (default: all).")
    g.add_argument("--placebo-reps", type=int, default=3,
                   help="Within-trader placebo permutations (default: 3).")

    g = parser.add_argument_group("outputs")
    g.add_argument("--results-dir", type=Path, default=config.RESULTS_DIR,
                   help="Where results are written (default: results/).")
    g.add_argument("--figures-dir", type=Path, default=config.FIGURES_DIR,
                   help="Where figures are written (default: reports/figures/).")

    args = parser.parse_args(argv)
    if args.force_refit:
        args.refit_models = args.reload_data = True
    return args


def setup_logging(level: int = logging.INFO) -> None:
    logging.basicConfig(
        level=level,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    )


class Outputs:
    """Writes result files and records which run wrote each one."""

    def __init__(self, directory: Path, args: argparse.Namespace):
        self.directory = Path(directory)
        self.written: List[str] = []
        try:
            sha = subprocess.run(["git", "rev-parse", "--short", "HEAD"],
                                 capture_output=True, text=True, cwd=config.PROJECT_ROOT,
                                 timeout=10).stdout.strip() or None
            dirty = subprocess.run(["git", "status", "--porcelain", "--untracked-files=no"],
                                   capture_output=True, text=True, cwd=config.PROJECT_ROOT,
                                   timeout=10).stdout.strip()
            if sha and dirty:
                sha += "-dirty"         # results came from uncommitted code
        except Exception:
            sha = None
        self.run_info = {
            "run_started": datetime.now().isoformat(timespec="seconds"),
            "package_version": __version__,
            "git_commit": sha,
            "command": " ".join(sys.argv),
            "libraries": cache.env_signature(),
            "subsample_traders": args.subsample_traders,
            "robustness_max_traders": args.robustness_max_traders,
        }

    def json(self, obj: Any, name: str) -> None:
        save_json(obj, name, self.directory)
        self.written.append(name)

    def csv(self, df: pd.DataFrame, name: str) -> None:
        save_csv(df, name, self.directory)
        self.written.append(name)

    def close(self, stages: Dict[str, bool]) -> None:
        update_manifest(self.directory, self.written, {**self.run_info, "stages": stages})


# ════════════════════════════════════════════════════════════════════════════
# Stage 1: data
# ════════════════════════════════════════════════════════════════════════════

def load_panel(args: argparse.Namespace) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """Cleaned trader-week panel and cleaning statistics."""
    logger.info("══ STAGE 1: DATA ══")
    parquet_path, stats_path = config.PANEL_PARQUET, config.CLEANING_STATS_JSON

    if parquet_path.exists() and not args.reload_data:
        logger.info(f"Loading cleaned panel from {parquet_path}")
        df = load_processed(str(parquet_path))
        if stats_path.exists():
            cleaning_stats = json.loads(stats_path.read_text())
        else:
            cleaning_stats = {"final_rows": len(df),
                              "unique_traders": df["trader_id"].nunique()}
    else:
        df = load_raw_data(str(args.data_path))

        miss = get_missingness_report(df)
        print("\n=== MISSINGNESS REPORT (RAW DATA) ===")
        print(miss[miss["n_missing"] > 0].to_string(index=False))
        print()
        logger.info(f"Column diagnostics: {verify_column_diagnostics(df)}")

        df, cleaning_stats = clean_data(df, apply_keep_country=None)
        print_cleaning_summary(cleaning_stats)
        logger.info(f"Verification checks: {verify_counts(cleaning_stats)}")

        save_processed(df, str(parquet_path))
        stats_path.parent.mkdir(parents=True, exist_ok=True)
        stats_path.write_text(json.dumps(
            {k: v if isinstance(v, (dict, list, str, int, float, bool)) else str(v)
             for k, v in cleaning_stats.items()},
            indent=2,
        ))

    logger.info(f"Working panel: {len(df):,} rows, {df['trader_id'].nunique():,} traders")

    # Subsample whole traders (row sampling would break the lags). Cleaning
    # statistics then describe the subsample, not the full raw file.
    if args.subsample_traders:
        ids = df["trader_id"].unique()
        if len(ids) > args.subsample_traders:
            rng = np.random.RandomState(config.RANDOM_STATE)
            keep = rng.choice(ids, size=args.subsample_traders, replace=False)
            df = df[df["trader_id"].isin(keep)].copy()
            logger.info(f"Subsampled to {args.subsample_traders:,} traders ({len(df):,} rows)")
            cleaning_stats = {"raw_rows": len(df), "final_rows": len(df),
                              "unique_traders": int(df["trader_id"].nunique()),
                              "subsample": True}

    return df, cleaning_stats


# ════════════════════════════════════════════════════════════════════════════
# Stage 2: analysis sample
# ════════════════════════════════════════════════════════════════════════════

@dataclass
class AnalysisSample:
    df: pd.DataFrame
    confounder_lag_cols: List[str]
    control_cols: List[str]
    country_dummy_cols: List[str]
    sample_counts: Dict[str, Any]
    flow_stats: Dict[str, Any]
    descriptives: Dict[str, Any]
    winsor_bounds: Dict[str, Any]
    audience_gaps: Dict[str, Any]
    activity: pd.DataFrame


def prepare_sample(df: pd.DataFrame, cleaning_stats: Dict[str, Any]) -> AnalysisSample:
    """Features, treatment/outcome variables, the analysis sample and its calendar."""
    logger.info("══ STAGE 2: FEATURES, VARIABLES, SAMPLE ══")
    window_start = pd.to_datetime(df["min_period_date"]).min()
    df = build_features(df, print_audit=True)
    audience_gaps = df.attrs.get("audience_gap_weeks", {})

    df, _ = compute_treatment(df)
    df = compute_outcome(df)
    winsor_bounds = df.attrs.get("winsor_bounds", {})

    # Lags and leads on the FULL panel (before sample filters), so the dynamic
    # check, the lead test and Anderson-Hsiao lose no rows
    df = df.sort_values(["trader_id", "period"])
    g = df.groupby("trader_id")
    df["total_gain_w_l1"] = g[OUTCOME].shift(1)
    df["total_self_trxn_f1"] = g[TREATMENT].shift(-1)
    df["total_self_trxn_l2"] = g[TREATMENT].shift(2)

    confounder_lag_cols = [f"{c}_l1" for c in CORE_LAG_COLUMNS]
    df_analysis, sample_counts = get_analysis_sample(
        df, treatment_col=TREATMENT, confounder_lag_cols=confounder_lag_cols,
        require_tenure=True,
    )
    logger.info(f"Analysis sample: {sample_counts}")

    flow_stats = {**cleaning_stats, **sample_counts,
                  "final_analysis_rows": len(df_analysis),
                  "rows_dropped_for_analysis":
                      cleaning_stats.get("final_rows", len(df)) - len(df_analysis)}

    if "user_country_name_bucketed" in df_analysis.columns:
        df_analysis, country_dummy_cols = get_country_dummies(df_analysis)
    else:
        country_dummy_cols = []

    descriptives = {**outcome_descriptives(df_analysis),
                    **activity_descriptives(df_analysis, window_start=window_start)}
    logger.info(f"Descriptives: {descriptives}")

    return AnalysisSample(
        df=df_analysis,
        confounder_lag_cols=confounder_lag_cols,
        # Tenure is (almost) absorbed by two-way FE; build_fe_arrays drops it
        # when it has no within variation
        control_cols=confounder_lag_cols + ["tenure_weeks"],
        country_dummy_cols=country_dummy_cols,
        sample_counts=sample_counts,
        flow_stats=flow_stats,
        descriptives=descriptives,
        winsor_bounds=winsor_bounds,
        audience_gaps=audience_gaps,
        activity=period_activity(df_analysis),
    )


# ════════════════════════════════════════════════════════════════════════════
# Stage 3: main models
# ════════════════════════════════════════════════════════════════════════════

@dataclass
class MainResults:
    models: List[Dict[str, Any]]
    primary: Dict[str, Any]
    arrays_primary: Dict[str, Any]
    arrays_pooled: Dict[str, Any]
    week_dec: pd.DataFrame
    week_blocks: pd.DataFrame
    aggregates: Dict[str, Any]
    regime: Dict[str, Any]


def estimate_main_models(s: AnalysisSample, refit: bool) -> MainResults:
    logger.info("══ STAGE 3: MAIN MODELS ══")
    df, controls = s.df, s.control_cols

    ols = naive_ols(Y=df[OUTCOME].values, T=df[TREATMENT].values,
                    groups=df["trader_id"].to_numpy())
    ols.update({"ate": ols["coef"], "ate_se": ols["se"],
                "ate_ci_lower": ols["ci_lower"], "ate_ci_upper": ols["ci_upper"]})

    def dml(arrays, cache_name, label, **kw):
        _, res = fit_linear_dml(
            Y=arrays["Y"], T=arrays["T"], W=arrays["W"],
            groups=arrays["groups"], time_ids=arrays["periods"],
            cache_name=cache_name, force_refit=refit, **kw,
        )
        res["model_name"] = label
        return res

    arrays_pooled = build_fe_arrays(df, TREATMENT, OUTCOME,
                                    controls + s.country_dummy_cols, fe="none")
    pooled = dml(arrays_pooled, "dml_pooled", "LinearDML, no FE (pooled)")

    arrays_fe1 = build_fe_arrays(df, TREATMENT, OUTCOME, controls, fe="trader")
    trader_fe = dml(arrays_fe1, "dml_trader_fe", "LinearDML, trader FE")

    arrays_fe2 = build_fe_arrays(df, TREATMENT, OUTCOME, controls, fe="twoway")
    primary = dml(arrays_fe2, "dml_twoway_fe", "LinearDML, trader + week FE (primary)",
                  return_residuals=True)
    Y_res, T_res = primary.pop("Y_res"), primary.pop("T_res")
    primary["controls"] = arrays_fe2["W_names"]

    ols_fe = linear_fe_ols(arrays_fe2)

    # Which weeks drive the primary estimate, and how it changes over time
    week_dec = week_decomposition(Y_res, T_res, arrays_fe2["periods"])
    week_dec = week_dec.merge(s.activity, on="period", how="left")
    aggregates = week_aggregates(week_dec, week_dec.set_index("period")["self_trades"])
    regime = regime_break(week_dec)
    week_blocks = week_block_estimates(df, controls, WEEK_BLOCKS)
    logger.info(f"Week aggregates: {aggregates}; regime break: {regime}")

    # Trader-level cross-section (one row per trader; between-trader question)
    df_trader = collapse_to_trader_level(df, confounder_cols=controls,
                                         treatment_col=TREATMENT, outcome_col=OUTCOME)
    valid_cols = [c for c in controls if c in df_trader.columns]
    df_trader = df_trader.dropna(subset=valid_cols + [TREATMENT, OUTCOME])
    cross_section = None
    if len(df_trader) > 100:
        a = prepare_model_arrays(df_trader, treatment_col=TREATMENT, outcome_col=OUTCOME,
                                 confounder_cols=valid_cols)
        _, cross_section = fit_linear_dml(Y=a["Y"], T=a["T"], W=a["W"],
                                          cache_name="dml_trader_level", force_refit=refit)
        cross_section["model_name"] = "LinearDML, trader-level cross-section"

    models = [m for m in [ols, pooled, trader_fe, primary, ols_fe, cross_section]
              if m is not None]
    return MainResults(models=models, primary=primary, arrays_primary=arrays_fe2,
                       arrays_pooled=arrays_pooled, week_dec=week_dec,
                       week_blocks=week_blocks, aggregates=aggregates, regime=regime)


# ════════════════════════════════════════════════════════════════════════════
# Stage 4: heterogeneity
# ════════════════════════════════════════════════════════════════════════════

@dataclass
class HeterogeneityResults:
    cate_heldout: np.ndarray
    moderators_heldout: pd.DataFrame
    gates: Dict[str, Any]
    summary: Dict[str, Any] = field(default_factory=dict)


def estimate_heterogeneity(s: AnalysisSample, m: MainResults,
                           args: argparse.Namespace) -> HeterogeneityResults:
    """Forest on a random half of traders; every test and figure on the held-out half."""
    logger.info("══ STAGE 4: HETEROGENEITY ══")
    arrays = m.arrays_primary
    X_df = s.df.loc[arrays["index"], HETEROGENEITY_COLS].astype(np.float64)
    X_all = X_df.to_numpy()
    groups, periods = arrays["groups"], arrays["periods"]

    rng = np.random.RandomState(config.HALF_SPLIT_SEED)
    uniq = np.unique(groups)
    in_a = np.isin(groups, rng.choice(uniq, size=len(uniq) // 2, replace=False))
    in_b = ~in_a

    cf, forest, _ = fit_causal_forest(
        Y=arrays["Y"][in_a], T=arrays["T"][in_a], X=X_all[in_a], W=arrays["W"][in_a],
        groups=groups[in_a], n_estimators=200, min_samples_leaf=50,
        max_samples=args.max_cf_samples, cache_name="causal_forest_twoway",
        force_refit=args.refit_models,
    )
    forest.pop("fit_row_index", None)
    # Summary statistics returned by fit_causal_forest describe the fitting
    # half (half A); rename them so they are not mistaken for held-out results
    forest = {(f"{k}_half_a" if k in {"ate_from_cate", "cate_mean", "cate_median", "cate_std",
                                      "cate_min", "cate_max", "share_positive",
                                      "share_negative"} else k): v
              for k, v in forest.items()}
    cate_b = cf.effect(X_all[in_b]).ravel()
    forest.update({
        "heldout_cate_mean": float(cate_b.mean()),
        "heldout_cate_median": float(np.median(cate_b)),
        "heldout_share_point_negative": float((cate_b < 0).mean()),
        "heldout_share_point_positive": float((cate_b > 0).mean()),
        "heterogeneity_features": HETEROGENEITY_COLS,
    })
    if "feature_importances" in forest:
        forest["feature_importances_named"] = dict(
            zip(HETEROGENEITY_COLS, forest["feature_importances"]))

    # Pointwise 95% intervals on held-out rows (econml's BLB intervals treat
    # rows as independent, so these are optimistic)
    idx_b = np.flatnonzero(in_b)
    pick = rng.choice(len(idx_b), size=min(20000, len(idx_b)), replace=False)
    lo, hi = cf.effect_interval(X_all[idx_b[pick]], alpha=0.05)
    lo, hi = np.asarray(lo).ravel(), np.asarray(hi).ravel()
    forest.update({
        "heldout_share_ci_below_zero": float(np.mean(hi < 0)),
        "heldout_share_ci_above_zero": float(np.mean(lo > 0)),
        "heldout_share_ci_covers_zero": float(np.mean((lo <= 0) & (hi >= 0))),
        "n_interval_rows": int(len(pick)),
    })

    # Held-out DML residuals (nuisances use W and X, as in the forest)
    W_b = np.column_stack([arrays["W"][in_b], X_all[in_b]])
    _, res_b = fit_linear_dml(
        Y=arrays["Y"][in_b], T=arrays["T"][in_b], W=W_b, groups=groups[in_b],
        cache_name="dml_twoway_heldout", force_refit=args.refit_models,
        return_residuals=True,
    )
    yb, tb, gb, pb = res_b["Y_res"], res_b["T_res"], groups[in_b], periods[in_b]
    blocks_b = week_block_label(pb)
    X_b = X_df[in_b].reset_index(drop=True)

    # 1. Does the forest ranking carry real heterogeneity? (BLP calibration, GATES)
    calibration = blp_calibration(cate_b, yb, tb, gb, time_ids=pb)
    gates = gates_test(cate_b, yb, tb, gb, block=blocks_b, time_ids=pb)
    T_raw_b = s.df.loc[arrays["index"][in_b], TREATMENT].to_numpy()
    q_b = pd.qcut(cate_b, 4, labels=False, duplicates="drop")
    for grp in gates["groups"]:
        q = q_b == grp["group"] - 1
        grp["share_self_trades_late_block"] = float(
            T_raw_b[q & (pb >= config.LATE_BLOCK_START)].sum() / max(T_raw_b[q].sum(), 1))
    logger.info(f"GATES: Wald p = {gates['wald_equal_pvalue']:.4g}; "
                f"BLP calibration b2 = {calibration['b2_het']:.3f}")

    # 2. Calendar vs trader heterogeneity: block effects, and trader-level
    #    moderators with block intercepts
    block_effects = group_effects(blocks_b, yb, tb, gb, order=list(WEEK_BLOCKS), time_ids=pb)
    moderators = blp_moderators(X_b[config.TRADER_MODERATORS], blocks_b, yb, tb, gb,
                                time_ids=pb)

    # 3. Held-out effects by join cohort (no forest involved)
    cohort = X_b["join_cohort_weeks"].to_numpy()
    c_bins = pd.qcut(cohort, 5, duplicates="drop")
    label = lambda iv: f"{iv.left:.0f} to {iv.right:.0f} wks"  # noqa: E731
    cohort_effects = group_effects(np.array([label(iv) for iv in c_bins]), yb, tb, gb,
                                   order=[label(iv) for iv in c_bins.categories], time_ids=pb)

    summary = {
        "causal_forest": forest,
        "blp_calibration_heldout": calibration,
        "gates_heldout": gates,
        "heldout_effects_by_week_block": block_effects,
        "blp_moderators_heldout": moderators,
        "heldout_effects_by_join_cohort": cohort_effects,
        "n_traders_half_a": int(len(np.unique(groups[in_a]))),
        "n_fit_traders": forest.get("n_fit_traders"),
        "n_heldout_traders": int(len(np.unique(gb))),
        "n_heldout_rows": int(in_b.sum()),
    }
    return HeterogeneityResults(cate_heldout=cate_b, moderators_heldout=X_b,
                                gates=gates, summary=summary)


# ════════════════════════════════════════════════════════════════════════════
# Stage 5: robustness
# ════════════════════════════════════════════════════════════════════════════

def run_robustness(s: AnalysisSample, m: MainResults, args: argparse.Namespace,
                   out: Outputs) -> pd.DataFrame:
    """Writes sensitivity, lead test, Anderson-Hsiao, overlap, grid, checks and placebo."""
    logger.info("══ STAGE 5: ROBUSTNESS ══")
    controls_no_tenure = [c for c in s.control_cols if c != "tenure_weeks"]

    out.json(sensitivity_analysis(m.arrays_primary, m.primary), "sensitivity.json")
    out.json(lead_test(s.df, controls_no_tenure), "lead_test.json")

    # Feedback-robust linear benchmark (sequential exogeneity)
    ah = {
        "instrument_T_lag2": anderson_hsiao(
            s.df, OUTCOME, "total_gain_w_l1", TREATMENT, "total_self_trxn_l1",
            "total_self_trxn_l2"),
        "instrument_T_lag1": anderson_hsiao(
            s.df, OUTCOME, "total_gain_w_l1", TREATMENT, "total_self_trxn_l1",
            "total_self_trxn_l1"),
        "note": "First differences with week effects; dT instrumented by an earlier "
                "level of T. Valid under sequential exogeneity; lag-2 instrument also "
                "allows contemporaneous dependence. No time-varying controls.",
    }
    out.json(ah, "feedback_robust.json")

    k = len(s.confounder_lag_cols)
    out.csv(overlap_check(m.arrays_pooled["T"], m.arrays_pooled["W"][:, :k],
                          feature_names=m.arrays_pooled["W_names"][:k]), "overlap.csv")

    grid = run_robustness_grid(s.df, s.control_cols, max_traders=args.robustness_max_traders)
    out.csv(grid, "robustness_grid.csv")

    out.csv(additional_checks(s.df, s.control_cols, s.country_dummy_cols, m.primary,
                              max_traders=args.robustness_max_traders),
            "robustness_checks.csv")

    out.csv(placebo_within_trader(s.df, controls_no_tenure, n_reps=args.placebo_reps,
                                  max_traders=args.robustness_max_traders), "placebo.csv")
    return grid


# ════════════════════════════════════════════════════════════════════════════
# Stage 6: outputs
# ════════════════════════════════════════════════════════════════════════════

def write_main_results(s: AnalysisSample, m: MainResults, args: argparse.Namespace,
                       out: Outputs) -> None:
    out.csv(m.week_dec, "week_decomposition.csv")
    out.csv(m.week_blocks, "week_blocks.csv")
    out.json({
        "primary_spec": {
            "estimator": "LinearDML (HistGBR nuisances, 5 cross-fitting folds that "
                         "never split a trader)",
            "fixed_effects": "trader + week (two-way demeaning)",
            "inference": "trader-clustered; week wild cluster bootstrap (Webb weights, "
                         f"{config.WILD_BOOTSTRAP_REPS} draws); trader x week two-way. "
                         f"Week-level inference not reported with fewer than "
                         f"{config.MIN_CLUSTERS_FOR_CRVE} week clusters.",
            "treatment": TREATMENT,
            "outcome": f"{OUTCOME} (total_gain winsorized at the 1st/99th "
                       f"percentiles of its non-zero values)",
            "controls_used": m.primary.get("controls"),
            "period_blocks": WEEK_BLOCKS,
        },
        "sample": {**s.sample_counts, "subsample_traders": args.subsample_traders},
        "descriptives": s.descriptives,
        "winsor_bounds_nonzero_1_99": s.winsor_bounds,
        "audience_recording_gap_weeks": s.audience_gaps,
        "week_aggregates": m.aggregates,
        "regime_break": m.regime,
        "models": m.models,
    }, "main_results.json")


def make_figures(s: AnalysisSample, m: MainResults,
                 h: Optional[HeterogeneityResults], grid: Optional[pd.DataFrame]) -> None:
    """Figures, numbered in the order they appear in the report."""
    logger.info("══ STAGE 6: FIGURES ══")
    figures.plot_sample_flow(s.flow_stats)                              # fig 1
    figures.plot_margin_distributions(s.df, variable=TREATMENT)         # fig 2
    figures.plot_causal_dag()                                           # fig 3
    figures.plot_forest_comparison(m.models)                            # fig 4
    figures.plot_week_decomposition(m.week_dec, m.week_blocks)          # fig 5
    if h is not None:
        figures.plot_cate_distribution(h.cate_heldout)                  # fig 6
        figures.plot_gates(h.gates)                                     # fig 7
        figures.plot_heterogeneity(                                     # fig 8
            cate=h.cate_heldout,
            tenure=h.moderators_heldout["join_cohort_weeks"].to_numpy(),
            audience=h.moderators_heldout["copiers_start"].to_numpy())
    if grid is not None and len(grid):
        figures.plot_spec_curve(grid)                                   # fig 9


def print_summary(m: MainResults, h: Optional[HeterogeneityResults]) -> None:
    print("\n" + "=" * 86)
    print("RESULTS SUMMARY  (effect of one extra self-trade per week on weekly gain)")
    print("=" * 86)
    for r in m.models:
        wild = (f"  week-wild [{r['ate_ci_lower_week_wild']:.2f}, {r['ate_ci_upper_week_wild']:.2f}]"
                f" p={r['ate_pvalue_week_wild']:.3f}"
                if r.get("ate_ci_lower_week_wild") is not None else "")
        print(f"  {r['model_name']:<45} {r['ate']:>9.4f}  "
              f"[{r['ate_ci_lower']:.4f}, {r['ate_ci_upper']:.4f}] {r.get('se_type', '')}{wild}")
    for _, r in m.week_blocks.iterrows():
        print(f"  Primary within {r['block']:<31} {r['ate']:>9.4f}  "
              f"[{r['ci_lower']:.4f}, {r['ci_upper']:.4f}]")
    print(f"  Week aggregates: {m.aggregates}")
    print(f"  Regime break: {m.regime}")
    if h is not None:
        print(f"  GATES (held-out) equal-effects p = {h.gates['wald_equal_pvalue']:.3g}")
    print("=" * 86 + "\n")


# ════════════════════════════════════════════════════════════════════════════
# Entry point
# ════════════════════════════════════════════════════════════════════════════

def main(argv: Optional[List[str]] = None) -> None:
    args = parse_args(argv)
    setup_logging()
    figures.set_figures_dir(args.figures_dir)
    cache.set_cache_dir(args.cache_dir)
    out = Outputs(args.results_dir, args)
    t0 = time.time()
    logger.info("CAUSAL ML PIPELINE: self-trading intensity → trader gains")

    df, cleaning_stats = load_panel(args)
    sample = prepare_sample(df, cleaning_stats)
    main_res = estimate_main_models(sample, refit=args.refit_models)
    write_main_results(sample, main_res, args, out)

    het = None
    if not args.skip_causal_forest:
        het = estimate_heterogeneity(sample, main_res, args)
        out.json(het.summary, "heterogeneity.json")

    grid = None
    if not args.skip_robustness:
        grid = run_robustness(sample, main_res, args, out)
    else:
        logger.info("Skipping robustness (--skip-robustness)")

    make_figures(sample, main_res, het, grid)
    out.close({"heterogeneity": het is not None, "robustness": grid is not None})
    logger.info(f"PIPELINE COMPLETE in {(time.time() - t0) / 60:.1f} minutes")
    print_summary(main_res, het)


if __name__ == "__main__":
    main()
