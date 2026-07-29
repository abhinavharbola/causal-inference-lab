"""
Section 1 diagnostics: covariate balance (standardized mean differences,
love plots) before and after matching, and overlap/common-support
reporting. These are the diagnostics the Section 4 LLM critique step
reads, so their output format is kept simple and tabular rather than
plot-only.

Common-support trimming logic itself lives in estimators.py
(apply_common_support_trim) and is imported here rather than duplicated;
this module reports on overlap, it doesn't re-decide the trimming rule.
"""

import logging

import numpy as np
import pandas as pd

from src.validation.estimators import apply_common_support_trim, get_matched_pairs

logger = logging.getLogger(__name__)

# Conventional imbalance threshold (Austin, 2011): |SMD| > 0.1 is
# considered meaningful imbalance for a covariate.
SMD_IMBALANCE_THRESHOLD = 0.1


def compute_smd(
    df: pd.DataFrame,
    covariate_cols: list,
    treatment_col: str,
) -> pd.DataFrame:
    """
    Standardized mean difference per covariate:
        SMD = (mean_treated - mean_control) / pooled_std
    where pooled_std = sqrt((var_treated + var_control) / 2).

    This is the standard covariate-balance metric independent of sample
    size, unlike a raw t-test which becomes significant on trivial
    differences at large n.
    """
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
    """
    Combines pre- and post-matching SMD into a single table, one row per
    covariate, for direct before/after comparison. This is the data
    backing the love plot.
    """
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
    """
    Renders a love plot (SMD before vs after matching, one row per
    covariate) using matplotlib. Returns the figure so the caller
    (notebook or Streamlit dashboard) decides whether to show or save it.
    """
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
    """
    Summarizes propensity score overlap between treated and control arms:
    per-group range, the overlapping region, and the share of the sample
    that falls outside common support under both the overlap rule and a
    fixed [0.1, 0.9] band. Reuses apply_common_support_trim from
    estimators.py so the trimming logic isn't duplicated, just reported on.
    """
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
) -> dict:
    """
    Convenience wrapper bundling balance and overlap diagnostics into one
    call. This is the object the Section 4 LLM critique step reads from,
    so its keys are kept flat and simple rather than deeply nested.

    Takes only the pre-matching dataframe (with propensity already
    attached) and builds the matched sample internally via
    get_matched_pairs, so "before" and "after" balance are computed on
    the correct pair of samples rather than requiring the caller to pass
    a matched dataframe constructed elsewhere.
    """
    matched = get_matched_pairs(df_before, treatment_col, propensity_col, caliper)

    balance = balance_table(df_before, matched, covariate_cols, treatment_col)
    overlap = overlap_diagnostics(matched, propensity_col, treatment_col)

    return {
        "balance_table": balance,
        "overlap": overlap,
        "matched_df": matched,
        "n_imbalanced_covariates": int(balance["still_imbalanced"].sum()),
        "pct_within_overlap": overlap["pct_within_overlap"],
    }