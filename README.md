# Does Self-Trading Intensity Affect Trader Gains?
### A Causal Machine Learning Analysis of a Social Trading Platform

[![Python](https://img.shields.io/badge/Python-3.11-blue?logo=python)](https://www.python.org/)
[![EconML](https://img.shields.io/badge/EconML-0.17-green)](https://github.com/py-why/EconML)
[![Tests](https://img.shields.io/badge/tests-54%20passing-brightgreen)]()

This repository contains the analysis code for a study of how **self-trading intensity** (self-initiated trades per week) affects **weekly trader gains** on the XTrader social trading platform. The data is a trader × week panel from 2013.

- **Primary estimator:** Double Machine Learning with trader and week fixed effects. Cross-fitting folds never split a trader, and inference is clustered.
- **Heterogeneity:** a causal forest, tested on held-out traders.
- **Diagnostics:** a robustness grid, a strict-exogeneity test, a week-by-week decomposition, a placebo test and a sensitivity analysis.

The data, results and written report are not part of this repository.

---

## Contents

1. [Repository layout](#repository-layout): what each folder and file is for, and why
2. [Setup](#setup)
3. [How to run](#how-to-run)
4. [Design decisions](#design-decisions)
5. [Citation](#citation)

---

## Repository layout

```
Social_Trading_DML/
│
├── README.md                    project overview, layout, how to run (this file)
├── pyproject.toml               package definition, dependencies, pytest configuration
├── requirements.txt             exact package versions of the tested environment
├── run_pipeline.py              one-command entry point for the full analysis
│
├── src/social_trading_dml/      the analysis, as an installable Python package
│   ├── config.py                paths, variable names, analysis constants
│   ├── pipeline.py              runs the six analysis stages; command-line options
│   ├── io.py                    writes JSON/CSV results
│   ├── data/                    raw data → analysis-ready panel
│   │   ├── loading.py
│   │   ├── features.py
│   │   └── variables.py
│   ├── estimation/              causal estimators
│   │   ├── panel.py
│   │   ├── dml.py
│   │   ├── inference.py
│   │   ├── dynamic.py
│   │   ├── heterogeneity.py
│   │   └── cache.py
│   ├── diagnostics/             how much to trust the estimates
│   │   ├── robustness.py
│   │   ├── identification.py
│   │   ├── sensitivity.py
│   │   └── common.py
│   └── viz/
│       └── figures.py           publication figures
│
├── tests/                       automated tests (pytest)
│   ├── synthetic.py             simulated panels with known answers
│   ├── test_data.py
│   ├── test_estimation.py
│   └── test_inference_and_diagnostics.py
│
└── notebooks/                   interactive views of the data and results
    ├── 01_eda.ipynb
    ├── 02_main_results.ipynb
    └── 03_robustness.ipynb
```

### Top-level files

| File | What it is | Why it is there |
|---|---|---|
| `run_pipeline.py` | A short launcher that puts `src/` on the path and calls `social_trading_dml.pipeline.main()` | Lets anyone run the full analysis from a fresh clone with `python run_pipeline.py`, without installing anything |
| `pyproject.toml` | Package metadata, flexible dependency ranges, the `social-trading-dml` command, and pytest settings | Makes the code installable (`pip install -e .`) and lets `pytest` work without path hacks |
| `requirements.txt` | The exact package versions the analysis was run with | Exact reproducibility of the results |

### `src/social_trading_dml/`: the analysis package

The code is a package, not a set of scripts, so every step can be imported, tested and reused. Modules are grouped by role (data → estimation → diagnostics → figures), and `pipeline.py` wires them together.

| Module | What it does | Why it is separate |
|---|---|---|
| `config.py` | Every path, the treatment and outcome variable names, the heterogeneity features, the calendar-week blocks and the random seed | One source of truth, so the pipeline, notebooks and tests cannot drift apart |
| `pipeline.py` | Runs the six stages in order: load data → build sample → main models → heterogeneity → robustness → figures. Defines the command-line options (`--help`) | Each stage is a named function with typed inputs and outputs, so the flow reads top to bottom and stages can be rerun on their own |
| `io.py` | Converts numpy types and writes JSON/CSV results | Keeps results machine-readable and consistently formatted |

#### `data/`: from the raw export to the analysis sample

| Module | What it does | Why |
|---|---|---|
| `loading.py` | Reads the raw CSV, converts dates, applies the `keep_country` filter, checks counts against the source paper, and caches the cleaned panel as parquet | Cleaning is slow and should happen once. The checks catch silent data changes |
| `features.py` | Repairs **recording gaps** in audience counts: weeks in which every trader shows zero links are set to missing and filled from adjacent weeks. Builds **lagged controls** per trader, **platform tenure** (weeks since joining), the pre-join flag, weeks present and country buckets | Controls must be measured before the treatment week, and lags are computed within trader so they never leak across traders |
| `variables.py` | Builds the treatment (`total_self_trxn`) and its variants. Builds the outcome: `total_gain`, winsorized at its own non-zero 1/99 percentiles, plus raw, asinh and self/copied components. Builds the analysis sample and descriptive statistics | Gains are zero in most weeks, so percentiles must be taken over non-zero values only. Each outcome variant supports a specific robustness or mechanism check |

#### `estimation/`: the causal estimators

| Module | What it does | Why |
|---|---|---|
| `panel.py` | Two-way (trader + week) demeaning by alternating projections, and first differences. `build_fe_arrays()` is the single function that turns a sample into Y, T, W, trader and week arrays for any fixed-effects structure | Fixed effects remove time-invariant trader traits (skill) and market-wide weekly shocks. Building every model's inputs in one place keeps specifications comparable |
| `dml.py` | `fit_linear_dml()`: Double Machine Learning with gradient-boosted nuisance models, cross-fitted on folds that keep each trader's history together (`trader_folds`). Every fit reports trader-clustered, week-level and trader × week inference. Also the naive OLS and linear fixed-effects benchmarks | DML removes regularisation bias from flexible controls. Trader-level folds stop a trader's other weeks leaking into their own predictions |
| `inference.py` | Wild cluster bootstrap by week (Webb weights; restricted p-value and the CI from inverting it) and the effective number of clusters | When the identifying variation sits in a few weeks, trader-clustered SEs are too small and analytic cluster SEs are unreliable; the wild bootstrap is the standard remedy |
| `dynamic.py` | Anderson–Hsiao estimator: first differences with week effects, the change in self-trades instrumented by an earlier level | Gains feed back into later self-trading, which biases fixed-effects estimators; this estimator only needs sequential exogeneity |
| `heterogeneity.py` | Causal forest for heterogeneous effects. On **held-out** traders: GATES, a best-linear-predictor calibration test, and a moderator test with period-block intercepts | Forest point predictions overstate heterogeneity, and calendar timing can masquerade as trader differences. Testing on traders the forest never saw, net of period blocks, separates the two |
| `cache.py` | Saves fitted models and residuals, keyed by a hash of the data, settings, learner definitions and library versions; identical fits are stored once | A full run takes about 40 minutes. With the cache, reruns take minutes and reproduce results exactly. Changes to estimation code that keep the inputs the same still need a `CACHE_VERSION` bump |

#### `diagnostics/`: how much to trust the estimates

| Module | What it does | Why |
|---|---|---|
| `robustness.py` | 24-specification grid (treatment × learner × sample × outcome). Single-change checks, each changing one design choice and showing the primary model on the same rows: fixed-effects structure, outcome components, crowding out of copy trades, trimming, first-difference and correlated-random-effects DML, lagged outcome. Within-trader placebo | Shows whether conclusions depend on arbitrary modelling choices |
| `identification.py` | Lead test for strict exogeneity. Decomposition of the estimate into week contributions with 95% CIs, three ways of averaging them, and a break test. Refits within period blocks. Overlap check | Shows where the identifying variation comes from and whether the key assumptions hold |
| `sensitivity.py` | Cinelli & Hazlett (2020) robustness values and benchmark bounds | Quantifies how strong an unobserved confounder would need to be to overturn a result |
| `common.py` | Fits one specification and summarises a result | Shared by the two modules above |

#### `viz/figures.py`

Draws every figure in a consistent, colour-blind-safe journal style (300 dpi PNG plus vector PDF). `set_figures_dir()` redirects output for test runs.

### `tests/`

| File | Tests |
|---|---|
| `synthetic.py` | Generators for test data: a small panel with the real column layout, and a simulated panel with a **known effect** confounded by trader effects and week shocks |
| `test_data.py` | Lags (values, first-period NaN, row count), tenure, no blanket `dropna` anywhere in the package, excluded columns, sample construction |
| `test_estimation.py` | Two-way demeaning on balanced and unbalanced panels; winsorizing of zero-inflated outcomes; gap repair; pre-join handling; cache keys; **recovery of a known effect** (pooled DML must fail, two-way FE DML must succeed); folds never split a trader; clustered SEs; published Cinelli–Hazlett values and benchmark bounds; week decomposition and its per-week SEs; GATES; placebo |
| `test_inference_and_diagnostics.py` | Wild cluster bootstrap (estimate, effective clusters, **size under the null with common week shocks**); week inference withheld with few clusters; linear FE equals the full regression; **lead test detects feedback and Anderson–Hsiao recovers the effect under feedback**; BLP calibration; moderator test separates real moderators from calendar proxies; overlap with binary controls; break test and week aggregates; partial gap weeks; pre-window moderators; manifest |

These guard the logic that is easiest to get silently wrong in panel causal inference.

### `notebooks/`

| Notebook | Purpose |
|---|---|
| `01_eda.ipynb` | Data checks on the raw CSV: missingness, the keep_country investigation, panel balance, counts against the source paper |
| `02_main_results.ipynb` | Displays the main estimates, the week decomposition, the outcome decomposition and the heterogeneity analysis, with figures |
| `03_robustness.ipynb` | Displays the grid, single-change checks, lead test, placebo, sensitivity analysis and overlap check |

The notebooks are stored without outputs. Notebooks 02 and 03 never re-estimate anything: they display what the pipeline wrote, so run the pipeline first. They warn if result files come from different runs.

---

## Setup

The tested environment is Python 3.11 on Windows.

```bash
git clone https://github.com/t20071/Social_Trading_DML.git
cd Social_Trading_DML

conda create -n social_trading_dml -c defaults python=3.11 pyarrow=24.0.0 rpds-py=0.28.0 -y
conda activate social_trading_dml
pip install -r requirements.txt        # exact versions; keeps the conda pyarrow/rpds-py (same versions)
pip install -e . --no-deps              # optional: makes the package importable everywhere
```

> **Windows note:** some pip-built binaries (pyarrow, rpds-py) can be blocked by Smart App Control / Application Control. That is why the conda step installs them first, at the versions pinned in `requirements.txt`, so pip leaves them alone. If an import still fails with "DLL load failed … Application Control policy", reinstall that package from conda `defaults`, e.g. `conda install -c defaults rpds-py`.

**Data.** The dataset is not included in this repository. Place the raw export at `data/raw/trader_data_exl.csv`, or pass another location with `--data-path`. The pipeline creates the rest of `data/` itself: the cleaned panel and the model cache.

---

## How to run

### Full analysis

```bash
python run_pipeline.py
```

The first run takes about 40 minutes. Fitted models are cached locally, so reruns take 1–4 minutes and reproduce the numbers exactly. After `pip install -e .`, `python -m social_trading_dml` and `social-trading-dml` do the same.

**Outputs** are written locally and are not version-controlled:
- machine-readable results (JSON/CSV) go to `results/`;
- figures (PNG/PDF) go to `reports/figures/`.

Use `--results-dir` and `--figures-dir` to choose other locations.

### Options

```bash
python run_pipeline.py --help

# Quick test on a subsample, written to scratch/ so the main outputs are untouched
python run_pipeline.py --subsample-traders 3000 --robustness-max-traders 800 \
    --placebo-reps 1 --results-dir scratch/results --figures-dir scratch/figures
```

| Option | Default | Purpose |
|---|---|---|
| `--subsample-traders N` | all | keep N random traders (whole histories, so lags stay valid) |
| `--robustness-max-traders N` | all | run the grid, checks and placebo on N random traders |
| `--skip-robustness` | off | main models and heterogeneity only |
| `--skip-causal-forest` | off | skip the forest and held-out tests |
| `--placebo-reps N` | 3 | within-trader placebo permutations |
| `--max-cf-samples N` | 200000 | max rows (whole traders) used to fit the forest |
| `--refit-models` | off | ignore the model cache and refit |
| `--cache-dir PATH` | `data/cache/` | model cache location (use a scratch folder for test runs) |
| `--reload-data` | off | re-read and re-clean the raw CSV |
| `--force-refit` | off | both of the above |
| `--data-path PATH` | `data/raw/trader_data_exl.csv` | raw data elsewhere |
| `--results-dir`, `--figures-dir` | `results/`, `reports/figures/` | output locations |

### Tests

```bash
pytest                                                   # 54 tests
pytest tests/test_estimation.py -k recovers_effect -v    # a single test
```

The tests use simulated data only, so they run without the dataset.

### Notebooks

```bash
jupyter notebook notebooks/
```

Notebook 01 needs the raw data. Notebooks 02 and 03 need a completed pipeline run.

---

## Design decisions

| Decision | Reason |
|---|---|
| Two-way (trader + week) fixed effects | Removes stable trader traits and market-wide weekly shocks, as in the source paper (Mani et al.) |
| Cross-fitting folds by trader | Prevents a trader's own other weeks from informing their nuisance predictions |
| Week-level inference (wild cluster bootstrap by week) alongside trader clustering | Activity is concentrated in a few weeks, so trader-only clustering understates uncertainty |
| Feedback-robust benchmark (Anderson–Hsiao) | The lead test rejects strict exogeneity, which fixed-effects estimators require |
| Winsorizing at each variable's non-zero percentiles | Gains are zero in most weeks; percentiles over all weeks clip real P&L |
| Lagged outcome excluded from the primary controls | It adds Nickell bias under fixed effects; it is shown as a comparison check |
| Heterogeneity tested on held-out traders, with moderators fixed before the window | Forest point predictions are not evidence of heterogeneity, and moderators that respond to past trading are not valid effect modifiers |
| Week-by-week decomposition of the pooled estimate | Shows which weeks identify the estimate before it is interpreted |
| Results written to files; notebooks only read them | One source of truth, so the notebooks cannot disagree with the pipeline |
| Model cache keyed by a content hash | Fast, exact reruns; a fit is reused only when its data and settings are identical |

---

## Citation

```bibtex
@misc{selftrading_causalml_2026,
  title = {Does Self-Trading Intensity Affect Trader Gains?
           A Causal Machine Learning Analysis of Social Trading},
  year  = {2026},
  note  = {Double machine learning with two-way fixed effects on the XTrader
           social trading panel. Code: https://github.com/t20071/Social_Trading_DML}
}
```

Data source: Mani et al., XTrader social trading manuscript (unpublished).
