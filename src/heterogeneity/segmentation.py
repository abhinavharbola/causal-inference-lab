"""
Section 2/3: segments users either via a simple business-relevant
quantile split on a chosen covariate, or via unsupervised clustering on
pre-treatment covariates, then measures the treatment effect per segment.

Clustering exists specifically to answer "does response differ by
segment," not as a standalone unsupervised demo: compute_segment_effects
is the function that turns a segment assignment into the actual causal
question this project cares about.

Criteo's covariates (f0-f11) are anonymized dense floats, not literal
recency/frequency/value fields, so "business-relevant split" here means
a quantile split on a chosen covariate (e.g. the one most predictive of
CATE) rather than a literal RFM segmentation.
"""

import logging

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from statsmodels.stats.proportion import proportions_ztest

from src.utils.bootstrap import bootstrap_diff_in_means

logger = logging.getLogger(__name__)


def quantile_segments(
    df: pd.DataFrame,
    value_col: str,
    n_bins: int = 4,
    labels: list = None,
) -> pd.Series:
    """
    Business-style segmentation: splits units into n_bins quantile-based
    groups on a single chosen covariate. Simpler and more interpretable
    than clustering when a single covariate already captures most of the
    business-relevant variation (e.g. the covariate most correlated with
    CATE or with the outcome).
    """
    if labels is None:
        labels = [f"q{i+1}" for i in range(n_bins)]

    segment = pd.qcut(df[value_col], q=n_bins, labels=labels, duplicates="drop")
    return segment


def cluster_segments(
    df: pd.DataFrame,
    feature_cols: list,
    n_clusters: int = 4,
    random_state: int = None,
) -> pd.Series:
    """
    Unsupervised segmentation: standardizes the given pre-treatment
    covariates and runs KMeans. Returns cluster labels as strings
    ('cluster_0', 'cluster_1', ...) so segment labels are consistent in
    type with quantile_segments' output.
    """
    X = df[feature_cols].to_numpy()
    X_scaled = StandardScaler().fit_transform(X)

    kmeans = KMeans(n_clusters=n_clusters, random_state=random_state, n_init=10)
    labels = kmeans.fit_predict(X_scaled)

    logger.info("KMeans clustering: %d clusters on %d rows", n_clusters, len(df))

    return pd.Series([f"cluster_{i}" for i in labels], index=df.index, name="segment")


def compute_segment_effects(
    df: pd.DataFrame,
    segment_col: str,
    treatment_col: str,
    outcome_col: str,
    n_bootstrap: int = 1000,
    alpha: float = 0.05,
    random_state: int = None,
) -> pd.DataFrame:
    """
    For each segment, computes:
    - naive difference-in-means treatment effect
    - bootstrap CI (via bootstrap_diff_in_means, stratified within segment)
    - a two-proportion z-test p-value (feeds into Section 3's
      Benjamini-Hochberg correction across segments, done in evaluation.py)
    - segment sample size (n_segment) and per-arm sizes

    Also attaches mean estimated CATE per segment if a 'cate' column is
    present in df, as a cross-check against the naive per-segment estimate.
    """
    rows = []
    has_cate = "cate" in df.columns

    for segment_value, group in df.groupby(segment_col, observed=True):
        treated = group.loc[group[treatment_col] == 1, outcome_col].to_numpy()
        control = group.loc[group[treatment_col] == 0, outcome_col].to_numpy()

        if len(treated) < 2 or len(control) < 2:
            logger.warning(
                "Segment %s has too few units in one arm (n_treated=%d, n_control=%d), skipping",
                segment_value,
                len(treated),
                len(control),
            )
            continue

        boot_result = bootstrap_diff_in_means(
            treated, control, n_bootstrap=n_bootstrap, alpha=alpha, random_state=random_state
        )

        count = np.array([treated.sum(), control.sum()])
        nobs = np.array([len(treated), len(control)])
        _, p_value = proportions_ztest(count, nobs)

        row = {
            "segment": segment_value,
            "n_segment": len(group),
            "n_treated": len(treated),
            "n_control": len(control),
            "point_estimate": boot_result["point_estimate"],
            "ci_lower": boot_result["ci_lower"],
            "ci_upper": boot_result["ci_upper"],
            "p_value": p_value,
        }

        if has_cate:
            row["mean_predicted_cate"] = group["cate"].mean()

        rows.append(row)

    result_df = pd.DataFrame(rows)
    logger.info("Computed segment effects for %d segments", len(result_df))
    return result_df