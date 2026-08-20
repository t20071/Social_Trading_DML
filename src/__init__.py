"""
Causal ML analysis: Does self-trading intensity affect trader gains?

Package modules:
- data_loading: load, clean, verify the XTrader panel dataset
- features: lagged confounders, tenure, country bucketing
- treatment_outcome: treatment/outcome construction, demeaning, collapse
- models: naive OLS, LinearDML (ATE), CausalForestDML (CATE)
- robustness: full specification grid + diagnostic tests
- viz: journal-quality figures
"""

__version__ = "1.0.0"
