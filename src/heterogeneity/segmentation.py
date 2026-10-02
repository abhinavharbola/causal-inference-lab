import logging

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from statsmodels.stats.proportion import proportions_ztest

from src.utils.bootstrap import analytic_ci_diff_in_proportions, bootstrap_diff_in_means

logger = logging.getLogger(__name__)

CATE_SEPARATION_THRESHOLD = 0.05


def quantile_segments(
    df: pd.DataFrame,
    value_col: str,
    n_bins: int = 4,
    labels: list = None,
) -> pd.Series:
    codes = pd.qcut(df[value_col], q=n_bins, labels=False, duplicates="drop")
    n_actual = int(codes.max()) + 1

    if n_actual < n_bins:
        logger.warning(
            "Requested %d quantile bins but duplicate edges collapsed them to %d", n_bins, n_actual
        )

    if labels is None:
        labels = [f"q{i + 1}" for i in range(n_actual)]
    elif len(labels) < n_actual:
        raise ValueError(f"Got {len(labels)} labels for {n_actual} bins")

    segment = codes.map(lambda c: labels[int(c)] if pd.notna(c) else np.nan)
    return segment.rename(value_col + "_segment")


def cluster_segments(
    df: pd.DataFrame,
    feature_cols: list,
    n_clusters: int = 4,
    random_state: int = None,
) -> pd.Series:
    X = df[feature_cols].to_numpy()
    X_scaled = StandardScaler().fit_transform(X)

    kmeans = KMeans(n_clusters=n_clusters, random_state=random_state, n_init=10)
    labels = kmeans.fit_predict(X_scaled)

    logger.info("KMeans clustering: %d clusters on %d rows", n_clusters, len(df))

    return pd.Series([f"cluster_{i}" for i in labels], index=df.index, name="segment")


def segment_cate_separation(
    df: pd.DataFrame,
    segment_col: str,
    cate_col: str = "cate",
) -> dict:
    groups = [g[cate_col].to_numpy() for _, g in df.groupby(segment_col, observed=True)]

    if len(groups) < 2:
        return {
            "f_statistic": float("nan"),
            "p_value": float("nan"),
            "eta_squared": 0.0,
            "meaningfully_separated": False,
        }

    f_stat, p_value = stats.f_oneway(*groups)

    grand_mean = df[cate_col].mean()
    ss_between = sum(len(g) * (g.mean() - grand_mean) ** 2 for g in groups)
    ss_total = ((df[cate_col] - grand_mean) ** 2).sum()
    eta_squared = ss_between / ss_total if ss_total > 0 else 0.0

    meaningfully_separated = eta_squared >= CATE_SEPARATION_THRESHOLD

    if meaningfully_separated:
        logger.info(
            "Segments explain %.1f%% of predicted-CATE variance (eta^2=%.4f, F=%.2f, p=%.2e)",
            100 * eta_squared, eta_squared, f_stat, p_value,
        )
    else:
        logger.warning(
            "Segments explain only %.1f%% of predicted-CATE variance (eta^2=%.4f, threshold=%.2f). "
            "The segmentation does not track the model's predicted treatment-effect variation; "
            "observed per-segment effects remain valid randomized comparisons but are not "
            "evidence about what the CATE model learned.",
            100 * eta_squared, eta_squared, CATE_SEPARATION_THRESHOLD,
        )

    return {
        "f_statistic": float(f_stat),
        "p_value": float(p_value),
        "eta_squared": float(eta_squared),
        "meaningfully_separated": bool(meaningfully_separated),
    }


def segment_effect_heterogeneity(effects_df: pd.DataFrame) -> dict:
    effects = effects_df["point_estimate"].to_numpy(dtype=float)
    se = effects_df["se"].to_numpy(dtype=float)

    if len(effects) < 2 or np.any(se <= 0) or np.any(np.isnan(se)):
        return {"q_statistic": float("nan"), "df": max(len(effects) - 1, 0), "p_value": float("nan"), "i_squared": float("nan")}

    weights = 1.0 / se ** 2
    pooled = (weights * effects).sum() / weights.sum()
    q_stat = float((weights * (effects - pooled) ** 2).sum())
    dof = len(effects) - 1
    p_value = float(stats.chi2.sf(q_stat, dof))
    i_squared = max(0.0, (q_stat - dof) / q_stat) if q_stat > 0 else 0.0

    return {
        "q_statistic": q_stat,
        "df": dof,
        "p_value": p_value,
        "i_squared": float(i_squared),
    }


def compute_segment_effects(
    df: pd.DataFrame,
    segment_col: str,
    treatment_col: str,
    outcome_col: str,
    n_bootstrap: int = 1000,
    alpha: float = 0.05,
    random_state: int = None,
) -> pd.DataFrame:
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

        analytic = analytic_ci_diff_in_proportions(
            treated.mean(), len(treated), control.mean(), len(control), alpha=alpha
        )

        row = {
            "segment": segment_value,
            "n_segment": len(group),
            "n_treated": len(treated),
            "n_control": len(control),
            "point_estimate": boot_result["point_estimate"],
            "se": analytic["se"],
            "ci_lower": boot_result["ci_lower"],
            "ci_upper": boot_result["ci_upper"],
            "p_value": p_value,
        }

        if has_cate:
            row["mean_predicted_cate"] = group["cate"].mean()

        rows.append(row)

    result_df = pd.DataFrame(rows)
    logger.info("Computed segment effects for %d segments", len(result_df))

    if has_cate and len(result_df) >= 2:
        result_df.attrs["cate_separation"] = segment_cate_separation(df, segment_col, cate_col="cate")

    if len(result_df) >= 2:
        result_df.attrs["effect_heterogeneity"] = segment_effect_heterogeneity(result_df)

    return result_df
