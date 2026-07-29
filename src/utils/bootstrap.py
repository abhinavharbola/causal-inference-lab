import logging

import numpy as np
import pandas as pd
from scipy import stats

logger = logging.getLogger(__name__)


def analytic_ci_diff_in_proportions(
    p1: float,
    n1: int,
    p2: float,
    n2: int,
    alpha: float = 0.05,
) -> dict:
    se = np.sqrt(p1 * (1 - p1) / n1 + p2 * (1 - p2) / n2)
    z = stats.norm.ppf(1 - alpha / 2)
    diff = p1 - p2

    return {
        "point_estimate": diff,
        "ci_lower": diff - z * se,
        "ci_upper": diff + z * se,
        "se": se,
        "alpha": alpha,
        "method": "analytic_wald",
    }


def bootstrap_diff_in_means(
    treatment_outcomes: np.ndarray,
    control_outcomes: np.ndarray,
    n_bootstrap: int = 1000,
    alpha: float = 0.05,
    random_state: int = None,
) -> dict:
    rng = np.random.default_rng(random_state)
    n_treat = len(treatment_outcomes)
    n_control = len(control_outcomes)

    point_estimate = treatment_outcomes.mean() - control_outcomes.mean()

    boot_estimates = np.empty(n_bootstrap)
    for i in range(n_bootstrap):
        boot_treat = rng.choice(treatment_outcomes, size=n_treat, replace=True)
        boot_control = rng.choice(control_outcomes, size=n_control, replace=True)
        boot_estimates[i] = boot_treat.mean() - boot_control.mean()

    ci_lower = np.percentile(boot_estimates, 100 * alpha / 2)
    ci_upper = np.percentile(boot_estimates, 100 * (1 - alpha / 2))

    return {
        "point_estimate": point_estimate,
        "ci_lower": ci_lower,
        "ci_upper": ci_upper,
        "bootstrap_estimates": boot_estimates,
        "n_bootstrap": n_bootstrap,
        "alpha": alpha,
        "method": "bootstrap_percentile_stratified",
    }


def bootstrap_ci(
    df: pd.DataFrame,
    estimator_fn,
    n_bootstrap: int = 1000,
    alpha: float = 0.05,
    stratify_col: str = None,
    random_state: int = None,
) -> dict:
    rng = np.random.default_rng(random_state)
    n = len(df)

    point_estimate = estimator_fn(df)

    boot_estimates = np.empty(n_bootstrap)
    for i in range(n_bootstrap):
        if stratify_col is not None:
            parts = []
            for _, group in df.groupby(stratify_col):
                idx = rng.integers(0, len(group), size=len(group))
                parts.append(group.iloc[idx])
            boot_df = pd.concat(parts, ignore_index=True)
        else:
            idx = rng.integers(0, n, size=n)
            boot_df = df.iloc[idx]

        try:
            boot_estimates[i] = estimator_fn(boot_df)
        except Exception as exc:
            logger.warning("Bootstrap iteration %d failed (%s), using NaN", i, exc)
            boot_estimates[i] = np.nan

    valid_estimates = boot_estimates[~np.isnan(boot_estimates)]
    n_failed = n_bootstrap - len(valid_estimates)
    if n_failed > 0:
        logger.warning("%d of %d bootstrap iterations failed and were dropped", n_failed, n_bootstrap)

    ci_lower = np.percentile(valid_estimates, 100 * alpha / 2)
    ci_upper = np.percentile(valid_estimates, 100 * (1 - alpha / 2))

    return {
        "point_estimate": point_estimate,
        "ci_lower": ci_lower,
        "ci_upper": ci_upper,
        "bootstrap_estimates": boot_estimates,
        "n_bootstrap": n_bootstrap,
        "n_failed": n_failed,
        "alpha": alpha,
        "method": "bootstrap_percentile",
    }


def ci_excludes_value(ci_lower: float, ci_upper: float, value: float) -> bool:
    return value < ci_lower or value > ci_upper