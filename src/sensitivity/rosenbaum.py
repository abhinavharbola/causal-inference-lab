import logging

import numpy as np
import pandas as pd
from scipy import stats

logger = logging.getLogger(__name__)


def _split_pairs(matched_df: pd.DataFrame, outcome_col: str, treatment_col: str, pair_id_col: str = "_pair_id"):
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