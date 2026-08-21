# Does Self-Trading Intensity Affect Trader Gains?
### A Causal Machine Learning Analysis of Social Trading

[![Python](https://img.shields.io/badge/Python-3.10-blue?logo=python)](https://www.python.org/)
[![EconML](https://img.shields.io/badge/EconML-0.15+-green)](https://github.com/microsoft/EconML)
[![DoWhy](https://img.shields.io/badge/DoWhy-0.11+-orange)](https://github.com/py-why/dowhy)
[![License](https://img.shields.io/badge/License-Academic-lightgrey)]()

---

## Overview

This project investigates the **causal effect of self-trading intensity on trader financial gains** on social trading platforms. Applied to a panel of **1,196,100 trader-week observations** from 26,580 traders over 46 weeks, it uses **Double Machine Learning (DML)**, **within-trader fixed effects**, and **CausalForestDML** to answer:

> **Does self-trading causally hurt or help a trader's gains on a social trading platform?**

**Key Finding:** Each additional self-trade per week reduces trader gains by approximately **$0.59** (95% CI: [-0.69, -0.49]), robust across all 24 specification checks.

---

## Project Structure

```
.
├── src/                        # Core analysis modules
│   ├── __init__.py             # Package entry point and version (v1.0.0)
│   ├── data_loading.py         # Loads, cleans, and validates the XTrader CSV dataset
│   ├── features.py             # Builds lagged confounders, tenure, and country dummies
│   ├── treatment_outcome.py    # Constructs treatment/outcome variables and applies within-trader demeaning
│   ├── models.py               # Fits Naive OLS, LinearDML (ATE), and CausalForestDML (CATE)
│   ├── robustness.py           # Runs 24-specification robustness grid and DoWhy refutation tests
│   └── viz.py                  # Generates all journal-quality figures (300 dpi PNG + PDF)
│
├── notebooks/
│   ├── 01_eda.ipynb            # Exploratory data analysis: distributions, missingness, panel structure
│   ├── 02_main_results.ipynb   # Main causal estimates: OLS, LinearDML, CausalForest results
│   └── 03_robustness.ipynb     # Robustness grid, overlap diagnostics, and DoWhy refutations
│
├── figures/                    # Auto-generated output figures (PNG + PDF, 300 dpi)
│   ├── fig1_sample_flow.png    # Sample construction funnel diagram
│   ├── fig2_causal_dag.png     # Directed Acyclic Graph (DAG) of the causal structure
│   ├── fig3_forest_comparison.png   # OLS vs. DML vs. CausalForest ATE comparison
│   ├── fig4_cate_distribution.png   # Distribution of individual treatment effects (CATE)
│   ├── fig5_heterogeneity.png       # CATE vs. tenure, audience size, past gains
│   ├── fig6_feature_importances.png # CATE heterogeneity feature importances
│   └── fig7_spec_curve.png          # Specification curve across all 24 robustness checks
│
├── tests/
│   └── test_features.py        # Unit tests for lag correctness, row preservation, and tenure logic
│
├── run_pipeline.py             # End-to-end CLI runner: cleaning -> features -> models -> figures
├── requirements.txt            # Core Python dependencies
├── requirements_frozen.txt     # Fully pinned environment for exact reproducibility
├── requirements_f.txt          # Alternative frozen requirements
└── README.md                   # This file
```

---

## Research Design

### Dataset

| Attribute | Value |
|-----------|-------|
| **Platform** | XTrader social trading platform (Mani et al.) |
| **Raw rows** | 1,314,404 |
| **Analysis sample** | **1,196,100** trader-week observations |
| **Unique traders** | 26,580 |
| **Time period** | 46 weeks (Jan 1 - Nov 15, 2013) |
| **Panel structure** | Trader x Week |

### Treatment & Outcome

| Variable | Definition |
|----------|------------|
| **Treatment (T)** | `total_self_trxn` — number of self-executed trades in a given trader-week |
| **Outcome (Y)** | `total_gain_loss` — total financial gain/loss (USD) in a given trader-week |

### Identification Strategy

```
Step 1: Naive OLS          -> Biased baseline (no confounder control)
Step 2: LinearDML          -> Residualizes T and Y on confounders via gradient boosting
Step 3: LinearDML + FE     -> Adds within-trader fixed effects via demeaning
Step 4: CausalForestDML    -> Non-parametric heterogeneous treatment effects (CATE)
```

### Confounders Controlled

- `total_self_trxn_lag1` — lagged self-trade count (prior week)
- `total_gain_loss_lag1` — lagged gains (prior week)
- `audience_size` — number of copiers following the trader
- `total_trxn_lag1` — lagged total activity level
- `tenure` — weeks since the trader's first platform activity
- Country fixed effects via one-hot encoding

---

## Main Results

| Model | ATE Estimate | 95% CI | Notes |
|-------|-------------|--------|-------|
| Naive OLS | -0.56 | — | Confounded baseline |
| LinearDML | -0.60 | [-0.70, -0.50] | LightGBM nuisance learners |
| LinearDML + FE | **-0.5884** | **[-0.69, -0.49]** | Within-trader demeaned |
| CausalForestDML | -0.60 (median) | — | Individual-level CATE |

**Heterogeneity (CATE):**
- **64.6%** of trader-weeks show a negative treatment effect
- **35.4%** of trader-weeks show a positive treatment effect
- **Tenure** is the dominant moderator — **81.2% feature importance**
- Mid-career traders (4-6 months) suffer the largest penalty from self-trading

---

## Installation

### 1. Clone the repository

```bash
git clone https://github.com/t20071/Social_Trading_DML.git
cd Social_Trading_DML
```

### 2. Create & activate the conda environment

```bash
conda create -n social_trading_dml python=3.10 -y
conda activate social_trading_dml
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

> For exact reproducibility, use the pinned environment:
> ```bash
> pip install -r requirements_frozen.txt
> ```

---

## Running the Pipeline

```bash
# Full pipeline: cleaning -> features -> models -> robustness -> figures
python run_pipeline.py

# Skip robustness grid (faster, ~2 min)
python run_pipeline.py --skip-robustness

# Quick test on a subsample
python run_pipeline.py --subsample 50000

# Force re-fit all models (ignore cache)
python run_pipeline.py --force-refit
```

> Full pipeline runtime on the complete 1.2M observation dataset: ~5 minutes.

---

## Code File Reference

### `src/` — Analysis Modules

| File | One-line Description |
|------|----------------------|
| `src/__init__.py` | Package entry point; declares module list and version (`1.0.0`) |
| `src/data_loading.py` | Loads the raw XTrader CSV, applies `keep_country` filter, drops lag-missingness rows, and caches to Parquet |
| `src/features.py` | Engineers one-period lagged confounders, computes trader tenure, and one-hot encodes country buckets |
| `src/treatment_outcome.py` | Constructs the treatment (`total_self_trxn`), outcome (`total_gain_loss`), and applies within-trader demeaning for fixed-effects identification |
| `src/models.py` | Fits Naive OLS, LinearDML (ATE with LightGBM/HistGBR nuisance learners), and CausalForestDML (CATE) with model caching |
| `src/robustness.py` | Executes the 24-specification robustness grid (treatment type x learner x sample x country controls) plus DoWhy refutation tests and overlap checks |
| `src/viz.py` | Produces all 7 journal-quality figures (300 dpi PNG + vector PDF) with colorblind-safe palette and confidence intervals |

### Root-level Scripts

| File | One-line Description |
|------|----------------------|
| `run_pipeline.py` | CLI entry point that orchestrates all pipeline steps end-to-end with `--skip-robustness`, `--subsample`, and `--force-refit` flags |

### Notebooks

| File | One-line Description |
|------|----------------------|
| `notebooks/01_eda.ipynb` | Exploratory analysis of trader distributions, treatment/outcome skew, panel balance, and missingness patterns |
| `notebooks/02_main_results.ipynb` | Runs and displays the main OLS, LinearDML, and CausalForest estimates with formatted results tables |
| `notebooks/03_robustness.ipynb` | Visualises the 24-specification curve, overlap diagnostics, and DoWhy sensitivity results |

### Tests

| File | One-line Description |
|------|----------------------|
| `tests/test_features.py` | Unit tests verifying lag formula correctness, row-count preservation, and tenure computation logic |

---

## Notebooks

Run Jupyter from the project root:

```bash
jupyter notebook notebooks/
```

---

## Testing

```bash
pytest tests/ -v
```

---

## Key Dependencies

| Package | Version | Purpose |
|---------|---------|---------|
| `econml` | >= 0.15 | LinearDML and CausalForestDML |
| `dowhy` | >= 0.11 | Refutation tests for causal assumptions |
| `lightgbm` | >= 4.0 | Gradient-boosted nuisance learners |
| `scikit-learn` | >= 1.3 | HistGradientBoosting and ML utilities |
| `statsmodels` | >= 0.14 | Naive OLS and statistical diagnostics |
| `pandas` | >= 2.0 | Panel data manipulation |
| `matplotlib` | >= 3.8 | Figure rendering |
| `networkx` | >= 3.2 | Causal DAG visualization |

---

## Methodology Reference

| Method | Reference |
|--------|-----------|
| Double Machine Learning | Chernozhukov et al. (2018), *Econometrics Journal* |
| Causal Forests | Athey & Wager (2019), *PNAS* |
| Trading & Overconfidence | Barber & Odean (2000), *Journal of Finance* |
| Dataset | Mani et al., XTrader Social Trading Platform |

---

## Key Recommendations

1. **Platform designers** — Nudge traders with < 6 months tenure toward copy-trading; they incur the largest self-trading penalties.
2. **Traders** — Self-trading is on average loss-generating; consider relying more on copy-trading strategies early in your career.
3. **Regulators** — Mandate disclosure of individual self-trading performance so users can make informed decisions.

---

## Limitations

- Time-varying unobserved confounders remain even after within-trader demeaning
- SUTVA may be violated due to copy-trading network interference
- Results are specific to XTrader platform (2013); external validity requires caution
- Linear dose-response assumed for ATE; CausalForest partially relaxes this assumption

---

## Citation

```bibtex
@misc{selftrading_causalml_2026,
  title = {Does Self-Trading Intensity Affect Trader Gains?
           A Causal Machine Learning Analysis of Social Trading},
  year  = {2026},
  note  = {Double Machine Learning with within-trader fixed effects
           applied to the XTrader social trading platform dataset.
           Code: https://github.com/t20071/Social_Trading_DML}
}
```

---

*Pipeline completed: 2026-08-19 | Runtime: ~5.1 min | Sample: 1,196,100 trader-week observations across 26,580 traders*
