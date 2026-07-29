"""
Shared confidence-interval machinery for Section 1 (validation) and
Section 2 (per-segment CATE estimates).

Two distinct tools live here, not one, because they solve different
problems:

- `bootstrap_ci` / `bootstrap_diff_in_means`: for estimators without a
  clean closed-form variance (PSM, IPW, AIPW, per-segment CATE). Uses
  stratified resampling to preserve the treatment/control ratio within
  each bootstrap draw.
- `analytic_ci_diff_in_proportions`: for the full-dataset ground-truth
  ATE (n ~ 13M), where a Wald normal-approximation CI is already exact
  for practical purposes. Bootstrapping at that scale burns CPU/RAM for
  no statistical gain, so it is deliberately not used there.
"""

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
    """
    Wald CI for the difference in two proportions (p1 - p2), using the
    normal approximation. Appropriate for large n only (rule of thumb:
    n*p and n*(1-p) both > ~10 for each group). Intended for the
    full-dataset ground-truth ATE, not for smaller confounded subsamples.
    """
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
    """
    Stratified bootstrap CI for a simple difference-in-means ATE.
    Resamples treatment and control arms independently (with replacement,
    same size as original arm) on every iteration, which preserves the
    treatment/control ratio rather than resampling the pooled data and
    risking a draw with a distorted ratio.
    """
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
    """
    General-purpose bootstrap CI for any estimator that takes a dataframe
    and returns a scalar point estimate. Used for PSM/IPW/AIPW and
    per-segment CATE, where the estimator itself is a black box from the
    CI machinery's point of view.

    If stratify_col is given (typically the treatment indicator), each
    bootstrap draw resamples within each stratum separately and
    concatenates, preserving stratum proportions. Otherwise resamples the
    whole dataframe with replacement.
    """
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
    """Convenience check used by the confounding validation gate."""
    return value < ci_lower or value > ci_upper