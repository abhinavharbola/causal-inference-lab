"""
Section 2 model-quality check + Section 3 statistical rigor layer.

Qini evaluation uses sklift.metrics (already a project dependency via the
fetch_criteo fallback loader) rather than a hand-rolled implementation:
qini_curve and qini_auc_score are the standard, well-tested reference
implementation for uplift model evaluation, so there's no reason to
reimplement them.

Benjamini-Hochberg correction uses statsmodels.stats.multitest, applied
across all segment-level p-values from segmentation.compute_segment_effects
before any per-segment effect is reported as statistically real.
"""

import logging

import pandas as pd
from sklift.metrics import qini_auc_score, qini_curve
from statsmodels.stats.multitest import multipletests

logger = logging.getLogger(__name__)


def evaluate_cate_qini(
    holdout_df: pd.DataFrame,
    cate_col: str,
    outcome_col: str,
    treatment_col: str,
) -> dict:
    """
    Evaluates T-learner CATE quality on a held-out split using the Qini
    coefficient. A T-learner's output should never be treated as correct
    just because it ran; this is the check that quantifies whether the
    predicted CATE actually ranks units by true uplift better than
    random targeting would.

    Returns the Qini coefficient (qini_auc_score) and the underlying
    curve points (qini_curve) for plotting.
    """
    y_true = holdout_df[outcome_col].to_numpy()
    uplift = holdout_df[cate_col].to_numpy()
    treatment = holdout_df[treatment_col].to_numpy()

    qini_coefficient = qini_auc_score(y_true=y_true, uplift=uplift, treatment=treatment)
    curve_x, curve_y = qini_curve(y_true=y_true, uplift=uplift, treatment=treatment)

    logger.info("Qini coefficient: %.5f", qini_coefficient)

    return {
        "qini_coefficient": qini_coefficient,
        "curve_x": curve_x,
        "curve_y": curve_y,
    }


def plot_qini(qini_result: dict, title: str = "Qini Curve"):
    """
    Renders the Qini curve using the points from evaluate_cate_qini,
    against the random-targeting diagonal baseline. Returns the figure
    for the caller to show or save.
    """
    import matplotlib.pyplot as plt

    curve_x = qini_result["curve_x"]
    curve_y = qini_result["curve_y"]

    fig, ax = plt.subplots(figsize=(6, 5))
    ax.plot(curve_x, curve_y, label=f"Model (Qini = {qini_result['qini_coefficient']:.4f})")
    ax.plot([curve_x[0], curve_x[-1]], [curve_y[0], curve_y[-1]], linestyle="--", color="gray", label="Random targeting")

    ax.set_xlabel("Number targeted")
    ax.set_ylabel("Cumulative incremental outcomes")
    ax.set_title(title)
    ax.legend()
    fig.tight_layout()

    return fig


def apply_benjamini_hochberg(
    segment_effects_df: pd.DataFrame,
    p_value_col: str = "p_value",
    alpha: float = 0.05,
) -> pd.DataFrame:
    """
    Applies Benjamini-Hochberg FDR correction across all segment-level
    p-values. Adds `p_value_adjusted` and `significant_after_correction`
    columns. This is the check that prevents "found a significant segment"
    from being a multiple-comparisons artifact when many segments are
    tested at once.
    """
    p_values = segment_effects_df[p_value_col].to_numpy()

    reject, p_adjusted, _, _ = multipletests(p_values, alpha=alpha, method="fdr_bh")

    result_df = segment_effects_df.copy()
    result_df["p_value_adjusted"] = p_adjusted
    result_df["significant_after_correction"] = reject

    n_significant_raw = (segment_effects_df[p_value_col] < alpha).sum()
    n_significant_adjusted = reject.sum()

    if n_significant_raw != n_significant_adjusted:
        logger.warning(
            "BH correction changed significance count: %d segments significant at raw alpha=%.3f, "
            "%d significant after correction",
            n_significant_raw,
            alpha,
            n_significant_adjusted,
        )

    return result_df