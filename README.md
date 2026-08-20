# Does Self-Trading Intensity Affect Trader Gains?
### A Causal Machine Learning Analysis of Social Trading

[![Python](https://img.shields.io/badge/Python-3.10-blue?logo=python)](https://www.python.org/)
[![EconML](https://img.shields.io/badge/EconML-0.15+-green)](https://github.com/microsoft/EconML)
[![DoWhy](https://img.shields.io/badge/DoWhy-0.11+-orange)](https://github.com/py-why/dowhy)
[![License](https://img.shields.io/badge/License-Academic-lightgrey)]()

---

## Overview

This project investigates the **causal effect of self-trading intensity on trader financial gains** in the context of social trading platforms. Using a rich trader-week panel dataset (~1.2M observations from 26,580 traders over 46 weeks), the study applies **Double Machine Learning (DML)**, **within-trader fixed effects**, and **CausalForestDML** to answer a fundamental question:

> **Does self-trading causally hurt or help a trader's gains on a social trading platform?**

**Key Finding:** Self-trading has a **statistically significant negative causal effect** on trader gains. Each additional self-trade in a given week reduces gains by approximately **$0.59** (95% CI: [-0.69, -0.49]), robust across all 24 specification checks.

---

## Project Structure

```
.
├── src/                        # Core analysis modules
│   ├── __init__.py             # Package metadata & version
│   ├── data_loading.py         # Data ingestion, cleaning & verification
│   ├── features.py             # Feature engineering & lag construction
│   ├── treatment_outcome.py    # Treatment/outcome construction & demeaning
│   ├── models.py               # OLS, LinearDML (ATE), CausalForestDML (CATE)
│   ├── robustness.py           # Robustness grid, overlap checks, DoWhy refutations
│   └── viz.py                  # Journal-quality figure generation
│
├── notebooks/
│   ├── 01_eda.ipynb            # Exploratory data analysis
│   ├── 02_main_results.ipynb   # Main causal estimates
│   └── 03_robustness.ipynb     # Robustness & sensitivity analysis
│
├── figures/                    # Generated figures (PNG + PDF, 300 dpi)
│   ├── fig1_sample_flow.png
│   ├── fig2_causal_dag.png
│   ├── fig3_forest_comparison.png
│   ├── fig4_cate_distribution.png
│   ├── fig5_heterogeneity.png
│   ├── fig6_feature_importances.png
│   └── fig7_spec_curve.png
│
├── data/
│   └── processed/              # Cleaned & processed datasets
│
├── tests/
│   └── test_features.py        # Unit tests for feature engineering
│
├── Documentation/              # Supplementary reports and presentations
│   ├── New_report_updated.pdf
│   ├── New_report_updated.docx
│   └── *.pptx                  # Presentation decks
│
├── run_pipeline.py             # Main CLI entry point (end-to-end pipeline)
├── convert_to_docx.py          # Utility: convert report.md to .docx
├── report.md                   # Full academic report (Markdown)
├── trader_data_exl.csv         # Raw dataset (~252 MB, 1,314,404 rows)
├── requirements.txt            # Core dependencies
├── requirements_frozen.txt     # Fully pinned/frozen environment
└── README.md                   # This file
```

---

## Research Design

### Data

| Attribute | Value |
|-----------|-------|
| **Source** | XTrader social trading platform (Mani et al.) |
| **Raw rows** | 1,314,404 |
| **Analysis sample** | **1,196,100** trader-week observations |
| **Unique traders** | 26,580 |
| **Time period** | 46 weeks (Jan 1 - Nov 15, 2013) |
| **Panel structure** | Trader x Week |

### Treatment & Outcome

| Variable | Definition |
|----------|------------|
| **Treatment** | `total_self_trxn` — number of self-executed trades in a given trader-week |
| **Outcome** | `total_gain_loss` — total financial gain/loss (USD) in a given trader-week |

### Causal Identification Strategy

```
1. Naive OLS          -> Baseline, endogeneity-biased estimate
2. LinearDML          -> Double ML: residualizes treatment & outcome on confounders
3. LinearDML (FE)     -> + within-trader fixed effects via demeaning
4. CausalForestDML    -> Non-parametric, heterogeneous treatment effects (CATE)
```

### Confounders Controlled

- Lagged self-trades (`total_self_trxn_lag1`)
- Lagged gains (`total_gain_loss_lag1`)
- Copier audience size (`audience_size`)
- Trader activity level (`total_trxn_lag1`)
- Platform tenure (weeks since first activity)
- Country fixed effects (via one-hot dummies)

---

## Main Results

| Model | ATE Estimate | 95% CI | Notes |
|-------|-------------|--------|-------|
| Naive OLS | -0.56 | — | Endogeneity-biased |
| LinearDML | -0.60 | [-0.70, -0.50] | Flexible nuisance learners |
| LinearDML (FE) | **-0.5884** | **[-0.69, -0.49]** | Within-trader demeaned |
| CausalForestDML | -0.60 (median) | — | Heterogeneous effects |

**Treatment Effect Heterogeneity (CATE):**
- **64.6%** of trader-weeks have **negative** treatment effects
- **35.4%** of trader-weeks have **positive** treatment effects
- **Tenure** is the dominant moderator (81.2% feature importance)
- Mid-career traders (4-6 months) suffer the largest losses from self-trading

---

## Installation

### 1. Clone the repository

```bash
git clone <repo-url>
cd "11. anti"
```

### 2. Create & activate the conda environment

```bash
conda create -p ./evn python=3.10 -y
conda activate ./evn
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

> **Note:** For a fully reproducible environment with pinned versions, use:
> ```bash
> pip install -r requirements_frozen.txt
> ```

---

## Running the Pipeline

The full end-to-end analysis is controlled by `run_pipeline.py`:

```bash
# Full pipeline (cleaning -> features -> models -> robustness -> figures -> report)
python run_pipeline.py

# Skip the robustness grid (faster, ~2 minutes)
python run_pipeline.py --skip-robustness

# Subsample for quick testing
python run_pipeline.py --subsample 50000

# Force re-fitting all models (ignore cached results)
python run_pipeline.py --force-refit
```

**Pipeline steps:**
1. **Data loading & cleaning** (`src/data_loading.py`)
2. **Feature engineering** (`src/features.py`) — lag construction, tenure, country dummies
3. **Treatment & outcome construction** (`src/treatment_outcome.py`) — demeaning, collapse
4. **Model fitting** (`src/models.py`) — OLS, LinearDML, CausalForestDML
5. **Robustness checks** (`src/robustness.py`) — 24-specification grid, DoWhy refutations
6. **Figure generation** (`src/viz.py`) — journal-quality outputs saved to `figures/`

> Full pipeline runtime: ~5 minutes on the complete 1.2M observation dataset.

---

## Notebooks

| Notebook | Description |
|----------|-------------|
| `01_eda.ipynb` | Exploratory data analysis: distributions, missing data, panel structure |
| `02_main_results.ipynb` | Main causal estimates: OLS, LinearDML, CausalForest |
| `03_robustness.ipynb` | Robustness grid, overlap diagnostics, DoWhy refutations |

---

## Testing

```bash
pytest tests/ -v
```

Tests cover feature engineering, lag construction, and treatment/outcome computation.

---

## Documentation & Outputs

| File | Description |
|------|-------------|
| `report.md` | Full academic report with methodology, results & discussion |
| `Documentation/New_report_updated.pdf` | PDF version of the full report |
| `Documentation/New_report_updated.docx` | Word version of the full report |
| `Documentation/*.pptx` | Presentation decks (Gamma, Gemini, Napkin AI versions) |

To regenerate the `.docx` from `report.md`:

```bash
python convert_to_docx.py
```

---

## Key Dependencies

| Package | Purpose |
|---------|---------|
| `econml >= 0.15` | LinearDML & CausalForestDML |
| `dowhy >= 0.11` | DoWhy refutation tests |
| `lightgbm >= 4.0` | Gradient-boosted nuisance learners |
| `scikit-learn >= 1.3` | ML utilities & HistGradientBoosting |
| `statsmodels >= 0.14` | OLS & statistical tests |
| `pandas >= 2.0` | Data manipulation |
| `matplotlib >= 3.8` | Visualization |
| `networkx >= 3.2` | Causal DAG visualization |

---

## Causal DAG

The identification strategy is grounded in a Directed Acyclic Graph (DAG) where confounders are explicitly modeled. See `figures/fig2_causal_dag.png` for the full visualization.

```
[Audience Size] --------------------------------------------> [Gains (Y)]
[Past Activity] ---------------------------------> [Gains (Y)]
[Past Gains]    ------------> [Confounders (W)] -> [Gains (Y)]
[Tenure]        --------------------------------> [Gains (Y)]
[Country]       -----------------------------> [Gains (Y)]

[Self-Trades (T)] -----------------------------------------> [Gains (Y)]
```

---

## Methodology Reference

| Method | Reference |
|--------|-----------|
| Double Machine Learning (DML) | Chernozhukov et al. (2018), *Econometrics Journal* |
| Causal Forests | Athey & Wager (2019), *PNAS* |
| Trading & Overconfidence | Barber & Odean (2000), *Journal of Finance* |
| Social Trading Data | Mani et al., XTrader Platform |

---

## Key Recommendations

Based on the causal analysis:

1. **Platform designers** — Implement nudge interventions for traders with < 6 months tenure to encourage copy-trading over self-trading.
2. **Traders** — Self-trading is, on average, a losing strategy. Allocate more activity to copy-trading, especially early in your career.
3. **Regulators** — Consider mandating disclosure of individual-level self-trading performance statistics to enable informed decision-making.

---

## Limitations

- Unobserved time-varying confounders remain even after within-trader demeaning
- SUTVA violations possible due to network effects on copy-trading
- External validity limited to XTrader platform, 2013 data
- Linear dose-response assumed for the ATE; CausalForest partially relaxes this

---

## Citation

If you use this code or findings in your research, please cite:

```bibtex
@misc{selftrading_causalml_2026,
  title   = {Does Self-Trading Intensity Affect Trader Gains?
             A Causal Machine Learning Analysis of Social Trading},
  year    = {2026},
  note    = {Double Machine Learning with within-trader fixed effects
             applied to the XTrader social trading platform dataset}
}
```

---

*Analysis completed: 2026-08-19 | Pipeline runtime: ~5.1 minutes | Dataset: 1,196,100 trader-week observations*
