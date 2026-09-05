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
    # Row identity, not covariate values, is what "each control used once" needs
    # to check. Criteo's anonymized f0-f11 columns are bucketed, so distinct
    # control rows routinely share identical covariate values; deduplicating on
    # covariate_cols would undercount n_control_unique and could flag correct
    # 1:1-without-replacement matching as broken. A synthetic id column survives
    # get_matched_pairs (it just carries through like any other column) and lets
    # us count actual distinct rows instead.
    id_col = "_diagnostics_row_id"
    df_before = df_before.copy()
    df_before[id_col] = np.arange(len(df_before))

    matched = get_matched_pairs(df_before, treatment_col, propensity_col, caliper, random_state=random_state)

    balance = balance_table(df_before, matched, covariate_cols, treatment_col)
    # Overlap must be measured on the pre-matching candidate pool, not on
    # `matched`. apply_common_support_trim's "overlap" method derives its trim
    # bounds from whatever population it's given, so running it on the matched
    # output is close to tautological: matched pairs were already selected for
    # being within a caliper of each other, so pct_within_overlap reads ~100%
    # almost regardless of how little the original treated/control populations
    # actually overlapped. Measuring it on df_before instead reports genuine
    # overlap in the candidate pool, which is what the LLM critique's "common
    # support overlap is strong/weak" flag is actually supposed to reflect.
    overlap = overlap_diagnostics(df_before, propensity_col, treatment_col)

    n_treated_total = int((df_before[treatment_col] == 1).sum())
    n_pairs = int((matched[treatment_col] == 1).sum())
    n_control_unique = int(matched.loc[matched[treatment_col] == 0, id_col].nunique())
    match_rate = n_pairs / n_treated_total if n_treated_total > 0 else 0.0

    matched = matched.drop(columns=[id_col])

    if n_control_unique != n_pairs:
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

