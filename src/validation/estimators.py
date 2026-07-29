import logging

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import NearestNeighbors
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
import statsmodels.api as sm

logger = logging.getLogger(__name__)


def naive_ols_ate(
    df: pd.DataFrame,
    outcome_col: str,
    treatment_col: str,
    covariate_cols: list = None,
) -> float:
    cols = [treatment_col] + (covariate_cols or [])
    X = sm.add_constant(df[cols])
    y = df[outcome_col]

    model = sm.OLS(y, X).fit()
    return model.params[treatment_col]


def fit_propensity_score(
    df: pd.DataFrame,
    treatment_col: str,
    covariate_cols: list,
    max_iter: int = 5000,
) -> np.ndarray:
    X = df[covariate_cols].to_numpy()
    T = df[treatment_col].to_numpy()

    model = make_pipeline(StandardScaler(), LogisticRegression(max_iter=max_iter))
    model.fit(X, T)
    propensity = model.predict_proba(X)[:, 1]

    propensity = np.clip(propensity, 1e-3, 1 - 1e-3)
    return propensity


def apply_common_support_trim(
    df: pd.DataFrame,
    propensity_col: str,
    treatment_col: str,
    method: str = "overlap",
    fixed_bounds: tuple = (0.1, 0.9),
) -> pd.DataFrame:
    treated_ps = df.loc[df[treatment_col] == 1, propensity_col]
    control_ps = df.loc[df[treatment_col] == 0, propensity_col]

    if method == "overlap":
        lower = max(treated_ps.min(), control_ps.min())
        upper = min(treated_ps.max(), control_ps.max())
    elif method == "fixed":
        lower, upper = fixed_bounds
    else:
        raise ValueError(f"Unknown method: {method}")

    mask = (df[propensity_col] >= lower) & (df[propensity_col] <= upper)
    n_dropped = (~mask).sum()
    if n_dropped > 0:
        logger.info(
            "Common support trim (%s, [%.4f, %.4f]): dropped %d of %d rows (%.1f%%)",
            method,
            lower,
            upper,
            n_dropped,
            len(df),
            100 * n_dropped / len(df),
        )
    return df.loc[mask].reset_index(drop=True)


def psm_ate(
    df: pd.DataFrame,
    outcome_col: str,
    treatment_col: str,
    propensity_col: str,
    caliper: float = 0.2,
) -> float:
    matched = get_matched_pairs(df, treatment_col, propensity_col, caliper)

    matched_treated_outcomes = matched.loc[matched[treatment_col] == 1, outcome_col].to_numpy()
    matched_control_outcomes = matched.loc[matched[treatment_col] == 0, outcome_col].to_numpy()

    return matched_treated_outcomes.mean() - matched_control_outcomes.mean()


def get_matched_pairs(
    df: pd.DataFrame,
    treatment_col: str,
    propensity_col: str,
    caliper: float = 0.2,
) -> pd.DataFrame:
    treated = df[df[treatment_col] == 1].reset_index(drop=True)
    control = df[df[treatment_col] == 0].reset_index(drop=True)

    ps_std = df[propensity_col].std()
    caliper_distance = caliper * ps_std

    nn = NearestNeighbors(n_neighbors=1)
    nn.fit(control[[propensity_col]].to_numpy())
    distances, indices = nn.kneighbors(treated[[propensity_col]].to_numpy())

    within_caliper = distances.ravel() <= caliper_distance
    n_unmatched = (~within_caliper).sum()
    if n_unmatched > 0:
        logger.info(
            "PSM: %d of %d treated units unmatched (no control within caliper)",
            n_unmatched,
            len(treated),
        )

    matched_treated = treated.loc[within_caliper].copy()
    matched_control = control.loc[indices.ravel()[within_caliper]].copy()

    pair_ids = np.arange(len(matched_treated))
    matched_treated["_pair_id"] = pair_ids
    matched_control["_pair_id"] = pair_ids

    return pd.concat([matched_treated, matched_control], ignore_index=True)


def ipw_ate(
    df: pd.DataFrame,
    outcome_col: str,
    treatment_col: str,
    propensity_col: str,
) -> float:
    T = df[treatment_col].to_numpy()
    Y = df[outcome_col].to_numpy()
    e = df[propensity_col].to_numpy()

    mu1 = (T * Y / e).sum() / (T / e).sum()
    mu0 = ((1 - T) * Y / (1 - e)).sum() / ((1 - T) / (1 - e)).sum()

    return mu1 - mu0


def aipw_ate(
    df: pd.DataFrame,
    outcome_col: str,
    treatment_col: str,
    covariate_cols: list,
    propensity_col: str,
    max_iter: int = 5000,
) -> float:
    T = df[treatment_col].to_numpy()
    Y = df[outcome_col].to_numpy()
    e = df[propensity_col].to_numpy()
    X = df[covariate_cols].to_numpy()

    model_treated = make_pipeline(StandardScaler(), LogisticRegression(max_iter=max_iter))
    model_treated.fit(X[T == 1], Y[T == 1])
    mu1 = model_treated.predict_proba(X)[:, 1]

    model_control = make_pipeline(StandardScaler(), LogisticRegression(max_iter=max_iter))
    model_control.fit(X[T == 0], Y[T == 0])
    mu0 = model_control.predict_proba(X)[:, 1]

    aipw_treated = mu1 + T * (Y - mu1) / e
    aipw_control = mu0 + (1 - T) * (Y - mu0) / (1 - e)

    return (aipw_treated - aipw_control).mean()


def run_estimator_comparison(
    df: pd.DataFrame,
    outcome_col: str,
    treatment_col: str,
    covariate_cols: list,
    caliper: float = 0.2,
    trim_method: str = "overlap",
) -> dict:
    propensity = fit_propensity_score(df, treatment_col, covariate_cols)
    df = df.copy()
    df["_propensity"] = propensity

    trimmed = apply_common_support_trim(df, "_propensity", treatment_col, method=trim_method)

    results = {
        "naive_ols": naive_ols_ate(df, outcome_col, treatment_col),
        "psm": psm_ate(trimmed, outcome_col, treatment_col, "_propensity", caliper=caliper),
        "ipw": ipw_ate(trimmed, outcome_col, treatment_col, "_propensity"),
        "aipw": aipw_ate(trimmed, outcome_col, treatment_col, covariate_cols, "_propensity"),
    }

    for method, estimate in results.items():
        logger.info("Estimator '%s': ATE = %.5f", method, estimate)

    return results