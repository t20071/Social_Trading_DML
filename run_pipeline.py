"""
run_pipeline.py — CLI entry point for the full Causal ML analysis pipeline.

Runs: cleaning → features → main models → robustness grid → figures → report
All steps use the conda/venv Python environment.

Usage:
    python run_pipeline.py
    python run_pipeline.py --skip-robustness
    python run_pipeline.py --subsample 50000
    python run_pipeline.py --force-refit
"""

import argparse
import logging
import os
import sys
import time
import json
from pathlib import Path

# Ensure src/ is importable
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np
import pandas as pd

from src.data_loading import (
    load_raw_data, clean_data, verify_counts, print_cleaning_summary,
    save_processed, load_processed, verify_column_diagnostics,
    get_missingness_report,
)
from src.features import (
    build_features, get_excluded_columns, CORE_LAG_COLUMNS,
    get_country_dummies, print_lag_audit_table,
)
from src.treatment_outcome import (
    compute_treatment, compute_outcome, demean_within_trader,
    collapse_to_trader_level, get_analysis_sample,
)
from src.models import (
    naive_ols, fit_linear_dml, fit_causal_forest, prepare_model_arrays,
)
from src.robustness import (
    run_robustness_grid, overlap_check, outlier_exclusion,
    heckman_selection, dowhy_refutation,
)
from src.viz import (
    plot_sample_flow, plot_causal_dag, plot_forest_comparison,
    plot_cate_distribution, plot_heterogeneity,
    plot_feature_importances, plot_spec_curve,
    plot_margin_distributions, set_journal_style,
)


def setup_logging(level=logging.INFO):
    logging.basicConfig(
        level=level,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    )


def parse_args():
    parser = argparse.ArgumentParser(
        description="Run the full Causal ML analysis pipeline."
    )
    parser.add_argument(
        "--skip-robustness", action="store_true",
        help="Skip the robustness specification grid (faster).",
    )
    parser.add_argument(
        "--subsample", type=int, default=None,
        help="Subsample N rows for faster iteration.",
    )
    parser.add_argument(
        "--force-refit", action="store_true",
        help="Force refitting models even if cached results exist.",
    )
    parser.add_argument(
        "--data-path", type=str, default="trader_data_exl.csv",
        help="Path to the raw CSV data file.",
    )
    parser.add_argument(
        "--skip-causal-forest", action="store_true",
        help="Skip CausalForestDML (expensive on full data).",
    )
    parser.add_argument(
        "--max-cf-samples", type=int, default=200000,
        help="Max samples for CausalForestDML fitting (default: 200000).",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    setup_logging()
    logger = logging.getLogger("pipeline")
    t0 = time.time()

    logger.info("=" * 70)
    logger.info("CAUSAL ML PIPELINE: Self-Trading → Trader Gains")
    logger.info("=" * 70)

    # ════════════════════════════════════════════════════════════════════
    # STAGE 1: DATA LOADING AND CLEANING
    # ════════════════════════════════════════════════════════════════════
    logger.info("\n══ STAGE 1: DATA LOADING ══")

    parquet_path = "data/processed/trader_panel.parquet"
    if os.path.exists(parquet_path) and not args.force_refit:
        logger.info("Loading cached cleaned data from parquet ...")
        df = load_processed(parquet_path)
        # Load cached stats if available
        stats_path = "data/processed/cleaning_stats.json"
        if os.path.exists(stats_path):
            with open(stats_path) as f:
                cleaning_stats = json.load(f)
        else:
            cleaning_stats = {"final_rows": len(df), "unique_traders": df["trader_id"].nunique()}
    else:
        df = load_raw_data(args.data_path)

        # Missingness report (before any cleaning)
        miss = get_missingness_report(df)
        print("\n=== MISSINGNESS REPORT (RAW DATA) ===")
        print(miss[miss["n_missing"] > 0].to_string(index=False))
        print()

        # Column diagnostics
        diag = verify_column_diagnostics(df)
        logger.info(f"Column diagnostics: {diag}")

        # Clean (with auto keep_country decision)
        df, cleaning_stats = clean_data(df, apply_keep_country=None)
        print_cleaning_summary(cleaning_stats)

        # Verify counts
        checks = verify_counts(cleaning_stats)
        logger.info(f"Verification checks: {checks}")

        # Save processed data
        save_processed(df, parquet_path)

        # Save stats
        stats_path = "data/processed/cleaning_stats.json"
        os.makedirs(os.path.dirname(stats_path), exist_ok=True)
        # Convert non-serializable types
        serializable_stats = {}
        for k, v in cleaning_stats.items():
            if isinstance(v, (dict, list, str, int, float, bool)):
                serializable_stats[k] = v
            else:
                serializable_stats[k] = str(v)
        with open(stats_path, "w") as f:
            json.dump(serializable_stats, f, indent=2)

    logger.info(f"Working dataset: {len(df):,} rows, {df['trader_id'].nunique():,} traders")

    # Optional subsample
    if args.subsample and len(df) > args.subsample:
        logger.info(f"Subsampling to {args.subsample:,} rows")
        df = df.sample(n=args.subsample, random_state=42)

    # ════════════════════════════════════════════════════════════════════
    # STAGE 2: FEATURE ENGINEERING
    # ════════════════════════════════════════════════════════════════════
    logger.info("\n══ STAGE 2: FEATURE ENGINEERING ══")

    df = build_features(df, print_audit=True)

    # ════════════════════════════════════════════════════════════════════
    # STAGE 3: TREATMENT & OUTCOME CONSTRUCTION
    # ════════════════════════════════════════════════════════════════════
    logger.info("\n══ STAGE 3: TREATMENT & OUTCOME ══")

    df, treatment_stats = compute_treatment(df)
    df = compute_outcome(df)

    # ════════════════════════════════════════════════════════════════════
    # STAGE 4: SAMPLE CONSTRUCTION
    # ════════════════════════════════════════════════════════════════════
    logger.info("\n══ STAGE 4: SAMPLE CONSTRUCTION ══")

    # Define confounder lag columns
    confounder_lag_cols = [f"{c}_l1" for c in CORE_LAG_COLUMNS]
    heterogeneity_cols = ["learner_count_l1", "total_follower_count_l1", "tenure_weeks"]

    # Get analysis sample (explicit narrow dropna per column)
    df_analysis, sample_counts = get_analysis_sample(
        df,
        treatment_col="total_self_trxn",
        confounder_lag_cols=confounder_lag_cols,
        require_tenure=True,
    )

    logger.info(f"Analysis sample: {sample_counts}")

    # Store sample flow stats for the figure
    flow_stats = {**cleaning_stats, **sample_counts}
    flow_stats["final_analysis_rows"] = len(df_analysis)
    flow_stats["rows_dropped_for_analysis"] = (
        cleaning_stats.get("final_rows", len(df)) - len(df_analysis)
    )

    # ════════════════════════════════════════════════════════════════════
    # STAGE 5: COUNTRY DUMMIES
    # ════════════════════════════════════════════════════════════════════
    logger.info("\n══ STAGE 5: COUNTRY DUMMIES ══")

    if "user_country_name_bucketed" in df_analysis.columns:
        df_analysis, country_dummy_cols = get_country_dummies(df_analysis)
        logger.info(f"Created {len(country_dummy_cols)} country dummies")
    else:
        country_dummy_cols = []

    # ════════════════════════════════════════════════════════════════════
    # STAGE 6: WITHIN-TRADER DEMEANING
    # ════════════════════════════════════════════════════════════════════
    logger.info("\n══ STAGE 6: DEMEANING ══")

    demean_cols = ["total_self_trxn", "total_gain_w"] + confounder_lag_cols
    df_analysis = demean_within_trader(df_analysis, demean_cols)

    # ════════════════════════════════════════════════════════════════════
    # STAGE 7: MAIN MODELS
    # ════════════════════════════════════════════════════════════════════
    logger.info("\n══ STAGE 7: MAIN MODELS ══")

    all_confounder_cols = confounder_lag_cols + ["tenure_weeks"]
    all_confounder_cols_with_country = all_confounder_cols + country_dummy_cols

    # --- 7a: Naive OLS ---
    logger.info("Fitting Naive OLS ...")
    ols_result = naive_ols(
        Y=df_analysis["total_gain_w"].values,
        T=df_analysis["total_self_trxn"].values,
    )

    # --- 7b: LinearDML (full panel, HistGBR) ---
    logger.info("Fitting LinearDML (full panel, HistGBR) ...")
    arrays = prepare_model_arrays(
        df_analysis,
        treatment_col="total_self_trxn",
        outcome_col="total_gain_w",
        confounder_cols=all_confounder_cols_with_country,
    )
    _, dml_full_result = fit_linear_dml(
        Y=arrays["Y"], T=arrays["T"], W=arrays["W"],
        nuisance_learner="histgbr",
        cache_name="dml_full_histgbr",
        force_refit=args.force_refit,
    )

    # --- 7c: LinearDML (demeaned, HistGBR) — primary FE spec ---
    logger.info("Fitting LinearDML (demeaned, HistGBR) ...")
    demeaned_confounder_cols = [f"{c}_demeaned" for c in confounder_lag_cols]
    # Only include demeaned cols that exist
    demeaned_confounder_cols = [
        c for c in demeaned_confounder_cols if c in df_analysis.columns
    ]
    arrays_dm = prepare_model_arrays(
        df_analysis,
        treatment_col="total_self_trxn_demeaned",
        outcome_col="total_gain_w_demeaned",
        confounder_cols=demeaned_confounder_cols,
    )
    _, dml_demeaned_result = fit_linear_dml(
        Y=arrays_dm["Y"], T=arrays_dm["T"], W=arrays_dm["W"],
        nuisance_learner="histgbr",
        cache_name="dml_demeaned_histgbr",
        force_refit=args.force_refit,
    )
    dml_demeaned_result["model_name"] = "LinearDML (within-trader demeaned, HistGBR)"

    # --- 7d: LinearDML (trader-level cross-section) ---
    logger.info("Fitting LinearDML (trader-level cross-section) ...")
    df_trader = collapse_to_trader_level(
        df_analysis,
        confounder_cols=all_confounder_cols,
        treatment_col="total_self_trxn",
        outcome_col="total_gain_w",
    )
    # Drop rows with NaN confounders (explicit narrow subset)
    valid_cols = [c for c in all_confounder_cols if c in df_trader.columns]
    df_trader_clean = df_trader.dropna(subset=valid_cols + ["total_self_trxn", "total_gain_w"])

    if len(df_trader_clean) > 100:
        arrays_tl = prepare_model_arrays(
            df_trader_clean,
            treatment_col="total_self_trxn",
            outcome_col="total_gain_w",
            confounder_cols=valid_cols,
        )
        _, dml_trader_result = fit_linear_dml(
            Y=arrays_tl["Y"], T=arrays_tl["T"], W=arrays_tl["W"],
            nuisance_learner="histgbr",
            cache_name="dml_trader_level_histgbr",
            force_refit=args.force_refit,
        )
        dml_trader_result["model_name"] = "LinearDML (trader-level cross-section)"
    else:
        logger.warning("Too few traders for trader-level DML")
        dml_trader_result = {
            "model_name": "LinearDML (trader-level)",
            "ate": np.nan, "ate_se": np.nan,
            "ate_ci_lower": np.nan, "ate_ci_upper": np.nan,
            "n": 0,
        }

    # --- 7e: CausalForestDML ---
    if not args.skip_causal_forest:
        logger.info("Fitting CausalForestDML ...")
        het_cols_available = [c for c in heterogeneity_cols if c in df_analysis.columns]
        # W for forest = confounders NOT in X
        w_cols_forest = [c for c in all_confounder_cols_with_country
                         if c not in het_cols_available]

        arrays_cf = prepare_model_arrays(
            df_analysis,
            treatment_col="total_self_trxn",
            outcome_col="total_gain_w",
            confounder_cols=w_cols_forest,
            heterogeneity_cols=het_cols_available,
        )

        _, cf_results, cate = fit_causal_forest(
            Y=arrays_cf["Y"], T=arrays_cf["T"],
            X=arrays_cf["X"], W=arrays_cf["W"],
            n_estimators=200,
            min_samples_leaf=50,
            max_samples=args.max_cf_samples,
            cache_name="causal_forest",
            force_refit=args.force_refit,
        )
    else:
        logger.info("Skipping CausalForestDML (--skip-causal-forest)")
        cf_results, cate = None, None

    # ════════════════════════════════════════════════════════════════════
    # STAGE 8: ROBUSTNESS
    # ════════════════════════════════════════════════════════════════════
    if not args.skip_robustness:
        logger.info("\n══ STAGE 8: ROBUSTNESS ══")

        robustness_df = run_robustness_grid(
            df_analysis,
            base_confounder_cols=all_confounder_cols,
            country_dummy_cols=country_dummy_cols,
            outcome_col="total_gain_w",
        )

        # Save robustness results
        robustness_df.to_csv("data/processed/robustness_results.csv", index=False)
        logger.info("Saved robustness results to data/processed/robustness_results.csv")

        # Overlap check
        logger.info("Running overlap/positivity check ...")
        overlap = overlap_check(
            T=arrays["T"], W=arrays["W"],
            feature_names=all_confounder_cols_with_country[:len(all_confounder_cols)],
        )

        # Outlier exclusion robustness
        logger.info("Running outlier exclusion check ...")
        df_no_outliers, outlier_stats = outlier_exclusion(df_analysis.copy())

        # Heckman selection
        logger.info("Running Heckman selection correction ...")
        heckman = heckman_selection(df, df_analysis)
        logger.info(f"Heckman results: {heckman}")

        # DoWhy refutation
        logger.info("Running DoWhy refutation tests ...")
        dowhy_results = dowhy_refutation(
            df_analysis,
            treatment_col="total_self_trxn",
            outcome_col="total_gain_w",
            confounder_cols=all_confounder_cols,
        )
        logger.info(f"DoWhy results: {dowhy_results}")
    else:
        logger.info("Skipping robustness (--skip-robustness)")
        robustness_df = None

    # ════════════════════════════════════════════════════════════════════
    # STAGE 9: FIGURES
    # ════════════════════════════════════════════════════════════════════
    logger.info("\n══ STAGE 9: FIGURES ══")

    # Fig 1: Sample flow
    plot_sample_flow(flow_stats)

    # Fig 2: Causal DAG
    plot_causal_dag()

    # Fig 3: Forest comparison
    results_for_forest = [ols_result, dml_full_result, dml_demeaned_result]
    if dml_trader_result.get("n", 0) > 0:
        results_for_forest.append(dml_trader_result)
    plot_forest_comparison(results_for_forest)

    # Figs 4-6: CATE (only if forest was fitted)
    if cate is not None:
        plot_cate_distribution(cate)
        plot_heterogeneity(
            cate=cate,
            tenure=df_analysis["tenure_weeks"].values,
            audience=df_analysis["learner_count_l1"].values,
        )
        if cf_results and "feature_importances" in cf_results:
            het_cols_available = [
                c for c in heterogeneity_cols if c in df_analysis.columns
            ]
            plot_feature_importances(
                np.array(cf_results["feature_importances"]),
                het_cols_available,
            )

    # Fig 7: Spec curve
    if robustness_df is not None:
        plot_spec_curve(robustness_df)

    # Diagnostic: Margin distributions
    plot_margin_distributions(df, variable="total_self_trxn")

    # ════════════════════════════════════════════════════════════════════
    # DONE
    # ════════════════════════════════════════════════════════════════════
    elapsed = time.time() - t0
    logger.info(f"\n{'=' * 70}")
    logger.info(f"PIPELINE COMPLETE in {elapsed / 60:.1f} minutes")
    logger.info(f"{'=' * 70}")

    # Print summary of results
    print("\n" + "=" * 70)
    print("RESULTS SUMMARY")
    print("=" * 70)
    print(f"  Naive OLS:             β = {ols_result['coef']:.4f}  "
          f"[{ols_result['ci_lower']:.4f}, {ols_result['ci_upper']:.4f}]")
    print(f"  LinearDML (full):      β = {dml_full_result['ate']:.4f}  "
          f"[{dml_full_result['ate_ci_lower']:.4f}, {dml_full_result['ate_ci_upper']:.4f}]")
    print(f"  LinearDML (demeaned):  β = {dml_demeaned_result['ate']:.4f}  "
          f"[{dml_demeaned_result['ate_ci_lower']:.4f}, {dml_demeaned_result['ate_ci_upper']:.4f}]")
    if dml_trader_result.get("n", 0) > 0:
        print(f"  LinearDML (trader):    β = {dml_trader_result['ate']:.4f}  "
              f"[{dml_trader_result['ate_ci_lower']:.4f}, {dml_trader_result['ate_ci_upper']:.4f}]")
    if cf_results:
        print(f"  CausalForestDML:       ATE = {cf_results['ate_from_cate']:.4f}  "
              f"share>0 = {cf_results['share_positive']:.1%}")
    print("=" * 70 + "\n")


if __name__ == "__main__":
    main()
