import logging

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold

from src.utils.bootstrap import analytic_ci_diff_in_proportions

logger = logging.getLogger(__name__)

PROPENSITY_CLIP = 1e-3


def standardized_mean_differences(
    df: pd.DataFrame,
    treatment_col: str,
    covariate_cols: list,
) -> pd.Series:
    treated_mask = df[treatment_col].to_numpy() == 1
    smd = {}
    for col in covariate_cols:
        x = df[col].to_numpy(dtype=np.float64)
        treated_x = x[treated_mask]
        control_x = x[~treated_mask]
        pooled_sd = np.sqrt((treated_x.var(ddof=1) + control_x.var(ddof=1)) / 2)
        smd[col] = (treated_x.mean() - control_x.mean()) / pooled_sd
    return pd.Series(smd)


def _fit_logistic(X: np.ndarray, y: np.ndarray, max_iter: int) -> LogisticRegression:
    model = LogisticRegression(max_iter=max_iter)
    model.fit(X, y)
    return model


def adjusted_ground_truth(
    df: pd.DataFrame,
    treatment_col: str,
    outcome_cols: list,
    covariate_cols: list,
    n_splits: int = 5,
    alpha: float = 0.05,
    max_iter: int = 1000,
    random_state: int = None,
) -> dict:
    T = df[treatment_col].to_numpy().astype(np.int8)
    n = len(T)

    X = df[covariate_cols].to_numpy(dtype=np.float32)
    X = (X - X.mean(axis=0)) / X.std(axis=0)

    outcomes = {col: df[col].to_numpy().astype(np.int8) for col in outcome_cols}
    propensity = np.empty(n)
    mu1 = {col: np.empty(n) for col in outcome_cols}
    mu0 = {col: np.empty(n) for col in outcome_cols}

    splitter = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state)
    for fold, (train_idx, hold_idx) in enumerate(splitter.split(np.zeros(n), T)):
        logger.info("Ground-truth cross-fitting: fold %d of %d", fold + 1, n_splits)

        propensity_model = _fit_logistic(X[train_idx], T[train_idx], max_iter)
        propensity[hold_idx] = propensity_model.predict_proba(X[hold_idx])[:, 1]

        treated_train = train_idx[T[train_idx] == 1]
        control_train = train_idx[T[train_idx] == 0]

        for col in outcome_cols:
            y = outcomes[col]
            mu1[col][hold_idx] = _fit_logistic(X[treated_train], y[treated_train], max_iter).predict_proba(X[hold_idx])[:, 1]
            mu0[col][hold_idx] = _fit_logistic(X[control_train], y[control_train], max_iter).predict_proba(X[hold_idx])[:, 1]

    propensity = np.clip(propensity, PROPENSITY_CLIP, 1 - PROPENSITY_CLIP)
    T_float = T.astype(np.float64)
    z = stats.norm.ppf(1 - alpha / 2)

    results = {}
    for col in outcome_cols:
        y = outcomes[col].astype(np.float64)

        scores = (
            (mu1[col] - mu0[col])
            + T_float * (y - mu1[col]) / propensity
            - (1 - T_float) * (y - mu0[col]) / (1 - propensity)
        )
        estimate = float(scores.mean())
        se = float(scores.std(ddof=1) / np.sqrt(n))

        ipw_treated = (T_float * y / propensity).sum() / (T_float / propensity).sum()
        ipw_control = ((1 - T_float) * y / (1 - propensity)).sum() / ((1 - T_float) / (1 - propensity)).sum()

        treated_y = y[T == 1]
        control_y = y[T == 0]
        naive = analytic_ci_diff_in_proportions(
            treated_y.mean(), len(treated_y), control_y.mean(), len(control_y), alpha=alpha
        )

        results[col] = {
            "point_estimate": estimate,
            "se": se,
            "ci_lower": estimate - z * se,
            "ci_upper": estimate + z * se,
            "ipw_estimate": float(ipw_treated - ipw_control),
            "naive_estimate": float(naive["point_estimate"]),
            "naive_ci_lower": float(naive["ci_lower"]),
            "naive_ci_upper": float(naive["ci_upper"]),
        }
        logger.info(
            "Adjusted ground truth for %s: %.5f [%.5f, %.5f] (naive %.5f, IPW %.5f)",
            col,
            estimate,
            results[col]["ci_lower"],
            results[col]["ci_upper"],
            results[col]["naive_estimate"],
            results[col]["ipw_estimate"],
        )

    return {
        "outcomes": results,
        "propensity_auc": float(roc_auc_score(T, propensity)),
        "n": int(n),
        "n_splits": int(n_splits),
        "method": "aipw_cross_fitted_full_data",
    }
