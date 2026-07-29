"""
Section 1: artificially confounds a subsample of the (randomized) Criteo
data by biasing which units are retained, so the validation pipeline can
be checked against a known ground-truth ATE.

Retention rule:
    retention_probability = sigmoid(g0 + g1*X + g2*X*T)

g2 controls confounding strength. Non-zero g2 makes retention depend on
the interaction of covariate and treatment arm, which is what actually
correlates X with T in the retained sample. A rule with g2=0 (or an X
uncorrelated with the outcome) would not reliably bias a treatment effect
estimate in randomized data, so two things are enforced here:

1. X is selected by correlation with the outcome, not chosen arbitrarily
   (see `select_confounding_covariate`). An outcome-irrelevant X can make
   the calibration loop chase a validation gate that never trips, no
   matter how high g2 goes.
2. A validation gate (see `check_confounding_validity`) verifies the
   induced confounding actually did something, before any estimator
   comparison runs on the retained sample.
"""

import logging

import numpy as np
import pandas as pd
from scipy import stats

from src.utils.bootstrap import ci_excludes_value

logger = logging.getLogger(__name__)


def select_confounding_covariate(
    df: pd.DataFrame,
    outcome_col: str,
    candidate_cols: list,
) -> str:
    """
    Picks the candidate covariate with the strongest absolute correlation
    to the outcome. This is a prerequisite for the retention rule to
    produce real confounding: if X doesn't correlate with the outcome,
    biasing retention by X*T interaction won't bias the treatment effect
    estimate, and the calibration loop below will never converge.
    """
    correlations = {}
    for col in candidate_cols:
        corr = df[col].corr(df[outcome_col])
        correlations[col] = corr

    best_col = max(correlations, key=lambda c: abs(correlations[c]))
    logger.info(
        "Selected confounding covariate: %s (corr with %s = %.4f)",
        best_col,
        outcome_col,
        correlations[best_col],
    )
    return best_col


def _sigmoid(z: np.ndarray) -> np.ndarray:
    return 1 / (1 + np.exp(-z))


def compute_retention_probability(
    X: np.ndarray,
    T: np.ndarray,
    g0: float,
    g1: float,
    g2: float,
) -> np.ndarray:
    """retention_probability = sigmoid(g0 + g1*X + g2*X*T)"""
    z = g0 + g1 * X + g2 * X * T
    return _sigmoid(z)


def induce_confounding(
    df: pd.DataFrame,
    x_col: str,
    treatment_col: str,
    g0: float,
    g1: float,
    g2: float,
    random_state: int = None,
) -> pd.DataFrame:
    """
    Samples a retained subsample from df according to the retention rule.
    X is standardized before computing retention probability so that g0/g1/g2
    have a consistent, interpretable scale regardless of the raw covariate's
    range.
    """
    rng = np.random.default_rng(random_state)

    X_raw = df[x_col].to_numpy()
    X = (X_raw - X_raw.mean()) / X_raw.std()
    T = df[treatment_col].to_numpy()

    retention_prob = compute_retention_probability(X, T, g0, g1, g2)
    keep_mask = rng.uniform(size=len(df)) < retention_prob

    retained = df.loc[keep_mask].reset_index(drop=True)
    logger.info(
        "Retention at g2=%.3f: kept %d of %d rows (%.1f%%)",
        g2,
        len(retained),
        len(df),
        100 * len(retained) / len(df),
    )
    return retained


def check_confounding_validity(
    retained_df: pd.DataFrame,
    x_col: str,
    treatment_col: str,
    outcome_col: str,
    ground_truth_ate: float,
    ground_truth_ci: tuple,
    corr_alpha: float = 0.05,
) -> dict:
    """
    Validation gate. Checks two conditions on the retained subsample:

    (a) correlation(X, T) is significantly nonzero (it should be ~0 in
        the untouched randomized data, so a nonzero correlation here
        confirms the retention rule actually broke randomization).
    (b) naive difference-in-means on the retained subsample falls outside
        the bootstrap/analytic CI of the ground-truth ATE (confirms the
        induced confounding actually biases the naive estimate, not just
        that X and T happen to correlate without moving the estimate).

    Both must hold for the confounding severity to be considered valid;
    if either fails, g2 needs to be increased and retention resampled.
    """
    X = retained_df[x_col].to_numpy()
    T = retained_df[treatment_col].to_numpy()

    corr_xt, corr_pvalue = stats.pearsonr(X, T)
    corr_significant = corr_pvalue < corr_alpha

    treat_outcomes = retained_df.loc[retained_df[treatment_col] == 1, outcome_col].to_numpy()
    control_outcomes = retained_df.loc[retained_df[treatment_col] == 0, outcome_col].to_numpy()
    naive_estimate = treat_outcomes.mean() - control_outcomes.mean()

    ci_lower, ci_upper = ground_truth_ci
    estimate_outside_ci = ci_excludes_value(ci_lower, ci_upper, naive_estimate)

    passes_gate = corr_significant and estimate_outside_ci

    return {
        "corr_xt": corr_xt,
        "corr_pvalue": corr_pvalue,
        "corr_significant": corr_significant,
        "naive_estimate": naive_estimate,
        "ground_truth_ate": ground_truth_ate,
        "ground_truth_ci": ground_truth_ci,
        "estimate_outside_ci": estimate_outside_ci,
        "passes_gate": passes_gate,
    }


def calibrate_confounding(
    df: pd.DataFrame,
    x_col: str,
    treatment_col: str,
    outcome_col: str,
    ground_truth_ate: float,
    ground_truth_ci: tuple,
    g0: float = 0.0,
    g1: float = 0.0,
    g2_init: float = 0.5,
    g2_step: float = 0.5,
    max_iters: int = 10,
    random_state: int = None,
) -> dict:
    """
    Calibration loop: starts at g2_init and increases g2 by g2_step until
    the validation gate passes or max_iters is reached. Returns the final
    g2, the retained subsample at that g2, and the full iteration history
    so a failed calibration is visible rather than silently accepted.

    Hard-capped at max_iters. If the gate never passes, this is treated
    as a build-time flag, not an infinite loop: the last attempt's
    diagnostics are returned along with converged=False so the caller can
    fall back to a manually chosen g2 grid instead of assuming convergence.
    """
    history = []
    g2 = g2_init
    last_retained = None
    last_g2 = g2_init

    for iteration in range(max_iters):
        retained = induce_confounding(df, x_col, treatment_col, g0, g1, g2, random_state)
        diagnostics = check_confounding_validity(
            retained, x_col, treatment_col, outcome_col, ground_truth_ate, ground_truth_ci
        )
        diagnostics["iteration"] = iteration
        diagnostics["g2"] = g2
        history.append(diagnostics)
        last_retained = retained
        last_g2 = g2

        logger.info(
            "Calibration iter %d: g2=%.3f, corr_xt=%.4f (p=%.4f), naive=%.5f, gate=%s",
            iteration,
            g2,
            diagnostics["corr_xt"],
            diagnostics["corr_pvalue"],
            diagnostics["naive_estimate"],
            diagnostics["passes_gate"],
        )

        if diagnostics["passes_gate"]:
            return {
                "converged": True,
                "g2": g2,
                "retained_df": retained,
                "history": history,
            }

        g2 += g2_step

    logger.warning(
        "Calibration did not converge after %d iterations (final g2=%.3f). "
        "Falling back to a manual g2 grid is recommended.",
        max_iters,
        last_g2,
    )
    return {
        "converged": False,
        "g2": last_g2,
        "retained_df": last_retained,
        "history": history,
    }


def run_dose_response_confounding(
    df: pd.DataFrame,
    x_col: str,
    treatment_col: str,
    outcome_col: str,
    ground_truth_ate: float,
    ground_truth_ci: tuple,
    severities: dict = None,
    g0: float = 0.0,
    g1: float = 0.0,
    random_state: int = None,
) -> dict:
    """
    Runs induce_confounding at several fixed g2 severities (dose-response
    design), rather than a single calibrated point. Default severities are
    illustrative starting points; the calibration loop above should be
    used first to find a g2 that reliably trips the validation gate, and
    that value can anchor the 'strong' end of this grid.
    """
    if severities is None:
        severities = {"none": 0.0, "mild": 0.5, "moderate": 1.5, "strong": 3.0}

    results = {}
    for label, g2 in severities.items():
        retained = induce_confounding(df, x_col, treatment_col, g0, g1, g2, random_state)
        diagnostics = check_confounding_validity(
            retained, x_col, treatment_col, outcome_col, ground_truth_ate, ground_truth_ci
        )
        results[label] = {
            "g2": g2,
            "retained_df": retained,
            "diagnostics": diagnostics,
        }
        logger.info(
            "Severity '%s' (g2=%.3f): naive=%.5f, gate_pass=%s",
            label,
            g2,
            diagnostics["naive_estimate"],
            diagnostics["passes_gate"],
        )

    return results