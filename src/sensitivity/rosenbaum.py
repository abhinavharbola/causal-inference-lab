"""
Section 4: Rosenbaum bounds sensitivity analysis.

Runs specifically on the PSM matched-pairs output (get_matched_pairs from
estimators.py), not on IPW/AIPW estimates. Rosenbaum bounds are defined
in terms of matched pairs and how discordant pair outcomes could be
explained by an unmeasured confounder of odds-ratio strength Gamma; there
is no equivalent matched-pair structure for a weighting-based estimator,
so applying this method to IPW/AIPW output would not be a standard or
well-defined analysis.

For binary outcomes (visit), this uses the McNemar-style Rosenbaum bounds:
among discordant matched pairs (pairs where treated and control outcomes
differ), the worst-case Gamma-adjusted p-value is computed under the two
extreme allocations of unmeasured bias, and the smallest Gamma at which
that worst-case p-value crosses alpha is the reported sensitivity bound.
"""

import logging

import numpy as np
import pandas as pd
from scipy import stats

logger = logging.getLogger(__name__)


def _split_pairs(matched_df: pd.DataFrame, outcome_col: str, treatment_col: str, pair_id_col: str = "_pair_id"):
    """
    Reshapes matched-pairs long format (one row per unit, linked by
    pair_id) into one row per pair with treated/control outcomes side by
    side, for discordance counting.
    """
    treated = matched_df[matched_df[treatment_col] == 1][[pair_id_col, outcome_col]]
    control = matched_df[matched_df[treatment_col] == 0][[pair_id_col, outcome_col]]

    merged = treated.merge(control, on=pair_id_col, suffixes=("_treated", "_control"))
    return merged


def count_discordant_pairs(
    matched_df: pd.DataFrame,
    outcome_col: str,
    treatment_col: str,
    pair_id_col: str = "_pair_id",
) -> dict:
    """
    Counts concordant and discordant matched pairs for a binary outcome.

    n_plus: pairs where treated=1, control=0 (treatment "wins")
    n_minus: pairs where treated=0, control=1 (control "wins")
    n_discordant = n_plus + n_minus
    """
    pairs = _split_pairs(matched_df, outcome_col, treatment_col, pair_id_col)

    y_t = pairs[f"{outcome_col}_treated"]
    y_c = pairs[f"{outcome_col}_control"]

    n_plus = int(((y_t == 1) & (y_c == 0)).sum())
    n_minus = int(((y_t == 0) & (y_c == 1)).sum())
    n_concordant = int(len(pairs) - n_plus - n_minus)

    return {
        "n_pairs": len(pairs),
        "n_concordant": n_concordant,
        "n_plus": n_plus,
        "n_minus": n_minus,
        "n_discordant": n_plus + n_minus,
    }


def rosenbaum_bound_at_gamma(n_plus: int, n_discordant: int, gamma: float, alternative: str = "greater") -> float:
    """
    Worst-case p-value at a given Gamma for the one-sided test that
    treatment increases the outcome (alternative='greater').

    Under Gamma, the probability a discordant pair favors treatment is
    bounded in [1/(1+Gamma), Gamma/(1+Gamma)]. The worst-case (most
    conservative, i.e. largest) p-value for 'greater' uses p = Gamma/(1+Gamma)
    as the null success probability in a one-sided binomial test.
    """
    if n_discordant == 0:
        return 1.0

    p_worst = gamma / (1 + gamma)

    if alternative == "greater":
        # P(X >= n_plus) under Binomial(n_discordant, p_worst)
        p_value = stats.binom.sf(n_plus - 1, n_discordant, p_worst)
    elif alternative == "less":
        p_value = stats.binom.cdf(n_plus, n_discordant, 1 - p_worst)
    else:
        raise ValueError(f"Unknown alternative: {alternative}")

    return p_value


def compute_rosenbaum_bounds(
    matched_df: pd.DataFrame,
    outcome_col: str,
    treatment_col: str,
    pair_id_col: str = "_pair_id",
    gamma_range: np.ndarray = None,
    alternative: str = "greater",
) -> pd.DataFrame:
    """
    Computes the worst-case p-value across a range of Gamma values. This
    is the table backing the Section 4 summary plot: as Gamma increases
    (more unmeasured confounding allowed), the worst-case p-value rises,
    and the point where it crosses alpha shows how much hidden bias would
    be needed to overturn the conclusion.
    """
    if gamma_range is None:
        gamma_range = np.arange(1.0, 5.05, 0.1)

    counts = count_discordant_pairs(matched_df, outcome_col, treatment_col, pair_id_col)
    n_plus = counts["n_plus"]
    n_discordant = counts["n_discordant"]

    rows = []
    for gamma in gamma_range:
        p_value = rosenbaum_bound_at_gamma(n_plus, n_discordant, gamma, alternative)
        rows.append({"gamma": gamma, "worst_case_p_value": p_value})

    bounds_df = pd.DataFrame(rows)
    bounds_df.attrs["n_pairs"] = counts["n_pairs"]
    bounds_df.attrs["n_discordant"] = n_discordant
    bounds_df.attrs["n_plus"] = n_plus
    bounds_df.attrs["n_minus"] = counts["n_minus"]

    return bounds_df


def find_critical_gamma(
    matched_df: pd.DataFrame,
    outcome_col: str,
    treatment_col: str,
    pair_id_col: str = "_pair_id",
    alpha: float = 0.05,
    gamma_max: float = 10.0,
    gamma_step: float = 0.05,
    alternative: str = "greater",
) -> dict:
    """
    Finds the smallest Gamma at which the worst-case p-value crosses
    alpha, i.e. the amount of unmeasured confounding (expressed as an
    odds-ratio) that would be needed to overturn significance at the
    matched-pairs estimate. Returns None for critical_gamma if the
    conclusion holds even at gamma_max (report this as "robust to at
    least gamma_max", not as an unbounded claim).
    """
    counts = count_discordant_pairs(matched_df, outcome_col, treatment_col, pair_id_col)
    n_plus = counts["n_plus"]
    n_discordant = counts["n_discordant"]

    gamma = 1.0
    critical_gamma = None

    while gamma <= gamma_max:
        p_value = rosenbaum_bound_at_gamma(n_plus, n_discordant, gamma, alternative)
        if p_value >= alpha:
            critical_gamma = gamma
            break
        gamma += gamma_step

    if critical_gamma is None:
        logger.info(
            "Conclusion robust to unmeasured confounding up to gamma_max=%.2f "
            "(worst-case p-value never crossed alpha=%.3f in this range)",
            gamma_max,
            alpha,
        )
    else:
        logger.info(
            "Critical gamma = %.2f: unmeasured confounding of this odds-ratio strength "
            "would be needed to overturn the conclusion at alpha=%.3f",
            critical_gamma,
            alpha,
        )

    return {
        "critical_gamma": critical_gamma,
        "gamma_max_checked": gamma_max,
        "alpha": alpha,
        "n_pairs": counts["n_pairs"],
        "n_discordant": n_discordant,
        "n_plus": n_plus,
        "n_minus": counts["n_minus"],
    }


def plot_rosenbaum_bounds(bounds_df: pd.DataFrame, alpha: float = 0.05, title: str = "Rosenbaum Sensitivity Bounds"):
    """
    Renders the Section 4 summary plot: worst-case p-value vs Gamma, with
    a horizontal reference line at alpha. Returns the figure for the
    caller (notebook or Streamlit) to show or save.
    """
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(bounds_df["gamma"], bounds_df["worst_case_p_value"], marker="o", markersize=3)
    ax.axhline(alpha, linestyle="--", color="red", linewidth=1, label=f"alpha = {alpha}")

    ax.set_xlabel("Gamma (unmeasured confounding strength)")
    ax.set_ylabel("Worst-case p-value")
    ax.set_title(title)
    ax.legend()
    fig.tight_layout()

    return fig


def run_rosenbaum_sensitivity(
    matched_df: pd.DataFrame,
    outcome_col: str,
    treatment_col: str,
    pair_id_col: str = "_pair_id",
    alpha: float = 0.05,
    gamma_max: float = 10.0,
) -> dict:
    """
    Convenience wrapper: computes the bounds table, the critical gamma,
    and the summary plot in one call, for the calibrated confounding
    severity's PSM matched-pairs output.
    """
    bounds_df = compute_rosenbaum_bounds(matched_df, outcome_col, treatment_col, pair_id_col)
    critical = find_critical_gamma(
        matched_df, outcome_col, treatment_col, pair_id_col, alpha=alpha, gamma_max=gamma_max
    )
    fig = plot_rosenbaum_bounds(bounds_df, alpha=alpha)

    return {
        "bounds_table": bounds_df,
        "critical_gamma_result": critical,
        "figure": fig,
    }