
"""Shared helpers: fit one specification, sample traders, summarise a result."""

import logging
from typing import Any, Dict, Optional

import numpy as np
import pandas as pd

from social_trading_dml.estimation.dml import fit_linear_dml

logger = logging.getLogger(__name__)


def fit_spec(arrays: Dict[str, Any], learner: str, cv: int, random_state: int,
              cache_name: Optional[str] = None,
              return_residuals: bool = False) -> Dict[str, Any]:
    _, res = fit_linear_dml(
        Y=arrays["Y"], T=arrays["T"], W=arrays["W"],
        nuisance_learner=learner, cv=cv, random_state=random_state,
        groups=arrays["groups"], time_ids=arrays.get("periods"),
        cache_name=cache_name, return_residuals=return_residuals,
    )
    return res


def sample_traders(df: pd.DataFrame, max_traders: Optional[int], random_state: int):
    if not max_traders:
        return df
    ids = df["trader_id"].unique()
    if len(ids) <= max_traders:
        return df
    rng = np.random.RandomState(random_state)
    keep = rng.choice(ids, size=max_traders, replace=False)
    logger.info(f"Robustness: using {max_traders:,} of {len(ids):,} traders")
    return df[df["trader_id"].isin(keep)]


def summarize_result(res: Dict[str, Any]) -> Dict[str, Any]:
    return {
        "ate": res.get("ate"),
        "se": res.get("ate_se"),
        "ci_lower": res.get("ate_ci_lower"),
        "ci_upper": res.get("ate_ci_upper"),
        "pvalue": res.get("ate_pvalue"),
        "pvalue_week_wild": res.get("ate_pvalue_week_wild"),
        "ci_lower_week_wild": res.get("ate_ci_lower_week_wild"),
        "ci_upper_week_wild": res.get("ate_ci_upper_week_wild"),
        "se_twoway": res.get("ate_se_twoway"),
        "ci_lower_twoway": res.get("ate_ci_lower_twoway"),
        "ci_upper_twoway": res.get("ate_ci_upper_twoway"),
        "n": res.get("n"),
        "n_traders": res.get("n_clusters"),
        "n_weeks": res.get("n_time_clusters"),
        "effective_weeks": res.get("effective_time_clusters"),
    }
