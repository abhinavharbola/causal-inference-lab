import logging

import numpy as np
import pandas as pd
from scipy import stats

from src.utils.bootstrap import analytic_ci_diff_in_proportions, ci_excludes_value

logger = logging.getLogger(__name__)

DEFAULT_SEVERITIES = {"none": 0.0, "mild": 0.5, "moderate": 1.0, "strong": 1.5}


def select_confounding_covariate(
    df: pd.DataFrame,
    outcome_col: str,
    candidate_cols: list,
) -> str:
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


def compute_retention_probabilities(
    X: np.ndarray,
    T: np.ndarray,
    g2: float,
    treated_share: float,
    keep_fraction: float,
) -> np.ndarray:
    if not 0 < treated_share < 1:
        raise ValueError("treated_share must lie strictly between 0 and 1")
    if not 0 < keep_fraction < 1:
        raise ValueError("keep_fraction must lie strictly between 0 and 1")

    lower = max(0.0, (keep_fraction - (1 - treated_share)) / treated_share)
    upper = min(1.0, keep_fraction / treated_share)

    treated_prob = lower + (upper - lower) * _sigmoid(g2 * X)
    control_prob = (keep_fraction - treated_share * treated_prob) / (1 - treated_share)

    treated_prob = np.clip(treated_prob, 0.0, 1.0)
    control_prob = np.clip(control_prob, 0.0, 1.0)

    return np.where(T == 1, treated_prob, control_prob)


def induce_confounding(
    df: pd.DataFrame,
    x_col: str,
    treatment_col: str,
    g2: float,
    keep_fraction: float = 0.5,
    random_state: int = None,
) -> pd.DataFrame:
    rng = np.random.default_rng(random_state)

    X_raw = df[x_col].to_numpy(dtype=float)
    x_std = X_raw.std()
    if x_std == 0:
        raise ValueError(f"Confounding covariate '{x_col}' has zero variance")
    X = (X_raw - X_raw.mean()) / x_std
    T = df[treatment_col].to_numpy()
    treated_share = float(T.mean())

    retention_prob = compute_retention_probabilities(X, T, g2, treated_share, keep_fraction)
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
    alpha: float = 0.05,
    corr_alpha: float = 0.05,
) -> dict:
    X = retained_df[x_col].to_numpy()
    T = retained_df[treatment_col].to_numpy()

    corr_xt, corr_pvalue = stats.pearsonr(X, T)
    corr_significant = bool(corr_pvalue < corr_alpha)

    treat_outcomes = retained_df.loc[retained_df[treatment_col] == 1, outcome_col].to_numpy()
    control_outcomes = retained_df.loc[retained_df[treatment_col] == 0, outcome_col].to_numpy()

    naive_ci = analytic_ci_diff_in_proportions(
        treat_outcomes.mean(),
        len(treat_outcomes),
        control_outcomes.mean(),
        len(control_outcomes),
        alpha=alpha,
    )
    naive_estimate = naive_ci["point_estimate"]

    truth_outside_naive_ci = bool(
        ci_excludes_value(naive_ci["ci_lower"], naive_ci["ci_upper"], ground_truth_ate)
    )

    passes_gate = corr_significant and truth_outside_naive_ci

    return {
        "corr_xt": corr_xt,
        "corr_pvalue": corr_pvalue,
        "corr_significant": corr_significant,
        "naive_estimate": naive_estimate,
        "naive_ci": (naive_ci["ci_lower"], naive_ci["ci_upper"]),
        "ground_truth_ate": ground_truth_ate,
        "ground_truth_outside_naive_ci": truth_outside_naive_ci,
        "passes_gate": bool(passes_gate),
    }


def calibrate_confounding(
    df: pd.DataFrame,
    x_col: str,
    treatment_col: str,
    outcome_col: str,
    ground_truth_ate: float,
    g2_init: float = 0.25,
    g2_step: float = 0.25,
    max_iters: int = 8,
    keep_fraction: float = 0.5,
    random_state: int = None,
) -> dict:
    history = []
    g2 = g2_init
    last_retained = None
    last_g2 = g2_init

    for iteration in range(max_iters):
        retained = induce_confounding(df, x_col, treatment_col, g2, keep_fraction, random_state)
        diagnostics = check_confounding_validity(
            retained, x_col, treatment_col, outcome_col, ground_truth_ate
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
        "Calibration did not converge after %d iterations (final g2=%.3f)",
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
    severities: dict = None,
    keep_fraction: float = 0.5,
    random_state: int = None,
) -> dict:
    if severities is None:
        severities = DEFAULT_SEVERITIES

    results = {}
    for label, g2 in severities.items():
        retained = induce_confounding(df, x_col, treatment_col, g2, keep_fraction, random_state)
        diagnostics = check_confounding_validity(
            retained, x_col, treatment_col, outcome_col, ground_truth_ate
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
