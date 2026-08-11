import logging

import numpy as np
import pandas as pd

from src.validation.estimators import apply_common_support_trim, get_matched_pairs

logger = logging.getLogger(__name__)

SMD_IMBALANCE_THRESHOLD = 0.1


def compute_smd(
    df: pd.DataFrame,
    covariate_cols: list,
    treatment_col: str,
) -> pd.DataFrame:
    treated = df[df[treatment_col] == 1]
    control = df[df[treatment_col] == 0]

    rows = []
    for col in covariate_cols:
        mean_t = treated[col].mean()
        mean_c = control[col].mean()
        var_t = treated[col].var()
        var_c = control[col].var()
        pooled_std = np.sqrt((var_t + var_c) / 2)

        smd = (mean_t - mean_c) / pooled_std if pooled_std > 0 else 0.0

        rows.append(
            {
                "covariate": col,
                "mean_treated": mean_t,
                "mean_control": mean_c,
                "smd": smd,
                "imbalanced": abs(smd) > SMD_IMBALANCE_THRESHOLD,
            }
        )

    return pd.DataFrame(rows)


def balance_table(
    df_before: pd.DataFrame,
    df_after: pd.DataFrame,
    covariate_cols: list,
    treatment_col: str,
) -> pd.DataFrame:
    smd_before = compute_smd(df_before, covariate_cols, treatment_col)
    smd_after = compute_smd(df_after, covariate_cols, treatment_col)

    merged = smd_before[["covariate", "smd"]].rename(columns={"smd": "smd_before"})
    merged = merged.merge(
        smd_after[["covariate", "smd"]].rename(columns={"smd": "smd_after"}),
        on="covariate",
    )
    merged["improved"] = merged["smd_after"].abs() < merged["smd_before"].abs()
    merged["still_imbalanced"] = merged["smd_after"].abs() > SMD_IMBALANCE_THRESHOLD

    n_still_imbalanced = merged["still_imbalanced"].sum()
    if n_still_imbalanced > 0:
        logger.warning(
            "%d of %d covariates remain imbalanced (|SMD| > %.2f) after matching",
            n_still_imbalanced,
            len(merged),
            SMD_IMBALANCE_THRESHOLD,
        )

    return merged


def love_plot(balance_df: pd.DataFrame, title: str = "Covariate Balance"):
    import matplotlib.pyplot as plt

    sorted_df = balance_df.sort_values("smd_before", key=abs)

    fig, ax = plt.subplots(figsize=(6, max(3, 0.35 * len(sorted_df))))
    y_pos = np.arange(len(sorted_df))

    ax.scatter(sorted_df["smd_before"].abs(), y_pos, label="Before matching", marker="o")
    ax.scatter(sorted_df["smd_after"].abs(), y_pos, label="After matching", marker="x")
    ax.axvline(SMD_IMBALANCE_THRESHOLD, linestyle="--", color="gray", linewidth=1)

    ax.set_yticks(y_pos)
    ax.set_yticklabels(sorted_df["covariate"])
    ax.set_xlabel("Absolute Standardized Mean Difference")
    ax.set_title(title)
    ax.legend()
    fig.tight_layout()

    return fig


def overlap_diagnostics(
    df: pd.DataFrame,
    propensity_col: str,
    treatment_col: str,
) -> dict:
    treated_ps = df.loc[df[treatment_col] == 1, propensity_col]
    control_ps = df.loc[df[treatment_col] == 0, propensity_col]

    overlap_lower = max(treated_ps.min(), control_ps.min())
    overlap_upper = min(treated_ps.max(), control_ps.max())

    trimmed_overlap = apply_common_support_trim(df, propensity_col, treatment_col, method="overlap")
    trimmed_fixed = apply_common_support_trim(df, propensity_col, treatment_col, method="fixed")

    result = {
        "treated_ps_range": (treated_ps.min(), treated_ps.max()),
        "control_ps_range": (control_ps.min(), control_ps.max()),
        "overlap_region": (overlap_lower, overlap_upper),
        "n_total": len(df),
        "n_within_overlap": len(trimmed_overlap),
        "pct_within_overlap": 100 * len(trimmed_overlap) / len(df),
        "n_within_fixed_band": len(trimmed_fixed),
        "pct_within_fixed_band": 100 * len(trimmed_fixed) / len(df),
    }

    logger.info(
        "Overlap diagnostics: %.1f%% of rows within overlap region [%.4f, %.4f]",
        result["pct_within_overlap"],
        overlap_lower,
        overlap_upper,
    )

    return result


def run_full_diagnostics(
    df_before: pd.DataFrame,
    covariate_cols: list,
    treatment_col: str,
    propensity_col: str,
    caliper: float = 0.2,
    random_state: int = None,
) -> dict:
    matched = get_matched_pairs(df_before, treatment_col, propensity_col, caliper, random_state=random_state)

    balance = balance_table(df_before, matched, covariate_cols, treatment_col)
    overlap = overlap_diagnostics(matched, propensity_col, treatment_col)

    n_treated_total = int((df_before[treatment_col] == 1).sum())
    n_pairs = int((matched[treatment_col] == 1).sum())
    n_control_unique = matched.loc[matched[treatment_col] == 0].drop_duplicates(subset=covariate_cols).shape[0]
    match_rate = n_pairs / n_treated_total if n_treated_total > 0 else 0.0

    if n_control_unique != n_pairs:
        # Matching is meant to be strictly 1:1 without replacement (see
        # get_matched_pairs). If a control row's covariates repeat across pairs
        # here, the matched sample is not what balance_table/overlap_diagnostics/
        # Rosenbaum bounds assume it is, so surface it loudly rather than silently
        # reporting a balance table for pairs that aren't really independent.
        logger.warning(
            "%d matched pairs but only %d distinct control rows were used — "
            "matching is not behaving as strict 1:1 without replacement.",
            n_pairs,
            n_control_unique,
        )

    logger.info(
        "Matching: %d of %d treated units matched (%.1f%%) against %d distinct controls",
        n_pairs,
        n_treated_total,
        100 * match_rate,
        n_control_unique,
    )

    return {
        "balance_table": balance,
        "overlap": overlap,
        "matched_df": matched,
        "n_imbalanced_covariates": int(balance["still_imbalanced"].sum()),
        "pct_within_overlap": overlap["pct_within_overlap"],
        "n_pairs": n_pairs,
        "n_treated_total": n_treated_total,
        "match_rate": match_rate,
        "n_control_unique": n_control_unique,
    }