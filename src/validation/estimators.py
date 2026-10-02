import logging

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sortedcontainers import SortedList
import statsmodels.api as sm

from src.utils.bootstrap import bootstrap_ci, bootstrap_mean_ci

logger = logging.getLogger(__name__)

PROPENSITY_COL = "_propensity"
PROPENSITY_CLIP = 1e-3


def _logit(p: np.ndarray) -> np.ndarray:
    p = np.clip(p, 1e-6, 1 - 1e-6)
    return np.log(p / (1 - p))


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

    return np.clip(propensity, PROPENSITY_CLIP, 1 - PROPENSITY_CLIP)


def fit_propensity_score_cross_fitted(
    df: pd.DataFrame,
    treatment_col: str,
    covariate_cols: list,
    n_splits: int = 5,
    max_iter: int = 5000,
    random_state: int = None,
) -> np.ndarray:
    X = df[covariate_cols].to_numpy()
    T = df[treatment_col].to_numpy()

    propensity = np.empty(len(df))
    splitter = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state)

    for train_idx, holdout_idx in splitter.split(X, T):
        model = make_pipeline(StandardScaler(), LogisticRegression(max_iter=max_iter))
        model.fit(X[train_idx], T[train_idx])
        propensity[holdout_idx] = model.predict_proba(X[holdout_idx])[:, 1]

    return np.clip(propensity, PROPENSITY_CLIP, 1 - PROPENSITY_CLIP)


def attach_propensity(
    df: pd.DataFrame,
    treatment_col: str,
    covariate_cols: list,
    cross_fit: bool = True,
    n_splits: int = 5,
    random_state: int = None,
    propensity_col: str = PROPENSITY_COL,
) -> pd.DataFrame:
    if cross_fit:
        propensity = fit_propensity_score_cross_fitted(
            df, treatment_col, covariate_cols, n_splits=n_splits, random_state=random_state,
        )
    else:
        propensity = fit_propensity_score(df, treatment_col, covariate_cols)

    out = df.copy()
    out[propensity_col] = propensity
    return out


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


def get_matched_pairs(
    df: pd.DataFrame,
    treatment_col: str,
    propensity_col: str,
    caliper: float = 0.2,
    random_state: int = None,
) -> pd.DataFrame:
    treated = df[df[treatment_col] == 1].reset_index(drop=True)
    control = df[df[treatment_col] == 0].reset_index(drop=True)

    treated_scores = _logit(treated[propensity_col].to_numpy())
    control_scores = _logit(control[propensity_col].to_numpy())

    caliper_distance = caliper * _logit(df[propensity_col].to_numpy()).std()

    rng = np.random.default_rng(random_state)
    match_order = rng.permutation(len(treated))

    available = SortedList(zip(control_scores.tolist(), range(len(control))))

    matched_treated_idx = []
    matched_control_idx = []

    for t_idx in match_order:
        if len(available) == 0:
            break

        t_score = treated_scores[t_idx]
        pos = available.bisect_left((t_score, -1))

        candidates = []
        if pos < len(available):
            candidates.append(available[pos])
        if pos > 0:
            candidates.append(available[pos - 1])

        best_score, best_idx = min(candidates, key=lambda c: abs(c[0] - t_score))
        if abs(best_score - t_score) <= caliper_distance:
            matched_treated_idx.append(t_idx)
            matched_control_idx.append(best_idx)
            available.remove((best_score, best_idx))

    n_unmatched = len(treated) - len(matched_treated_idx)
    if n_unmatched > 0:
        logger.info(
            "PSM (1:1, no replacement, logit caliper): %d of %d treated units unmatched "
            "(control pool exhausted or no control within caliper)",
            n_unmatched,
            len(treated),
        )

    matched_treated = treated.loc[matched_treated_idx].reset_index(drop=True).copy()
    matched_control = control.loc[matched_control_idx].reset_index(drop=True).copy()

    pair_ids = np.arange(len(matched_treated))
    matched_treated["_pair_id"] = pair_ids
    matched_control["_pair_id"] = pair_ids

    return pd.concat([matched_treated, matched_control], ignore_index=True)


def psm_ate(
    df: pd.DataFrame,
    outcome_col: str,
    treatment_col: str,
    propensity_col: str,
    caliper: float = 0.2,
    random_state: int = None,
) -> float:
    matched = get_matched_pairs(df, treatment_col, propensity_col, caliper, random_state=random_state)

    matched_treated_outcomes = matched.loc[matched[treatment_col] == 1, outcome_col].to_numpy()
    matched_control_outcomes = matched.loc[matched[treatment_col] == 0, outcome_col].to_numpy()

    return matched_treated_outcomes.mean() - matched_control_outcomes.mean()


def psm_pair_diffs(
    matched_df: pd.DataFrame,
    outcome_col: str,
    treatment_col: str,
    pair_id_col: str = "_pair_id",
) -> np.ndarray:
    treated = matched_df.loc[matched_df[treatment_col] == 1, [pair_id_col, outcome_col]]
    control = matched_df.loc[matched_df[treatment_col] == 0, [pair_id_col, outcome_col]]
    merged = treated.merge(control, on=pair_id_col, suffixes=("_treated", "_control"))
    return (merged[f"{outcome_col}_treated"] - merged[f"{outcome_col}_control"]).to_numpy()


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


def _outcome_model(max_iter: int):
    return make_pipeline(StandardScaler(), LogisticRegression(max_iter=max_iter))


def aipw_scores(
    df: pd.DataFrame,
    outcome_col: str,
    treatment_col: str,
    covariate_cols: list,
    propensity_col: str,
    max_iter: int = 5000,
) -> np.ndarray:
    T = df[treatment_col].to_numpy()
    Y = df[outcome_col].to_numpy()
    e = df[propensity_col].to_numpy()
    X = df[covariate_cols].to_numpy()

    model_treated = _outcome_model(max_iter)
    model_treated.fit(X[T == 1], Y[T == 1])
    mu1 = model_treated.predict_proba(X)[:, 1]

    model_control = _outcome_model(max_iter)
    model_control.fit(X[T == 0], Y[T == 0])
    mu0 = model_control.predict_proba(X)[:, 1]

    aipw_treated = mu1 + T * (Y - mu1) / e
    aipw_control = mu0 + (1 - T) * (Y - mu0) / (1 - e)

    return aipw_treated - aipw_control


def aipw_ate(
    df: pd.DataFrame,
    outcome_col: str,
    treatment_col: str,
    covariate_cols: list,
    propensity_col: str,
    max_iter: int = 5000,
) -> float:
    scores = aipw_scores(df, outcome_col, treatment_col, covariate_cols, propensity_col, max_iter=max_iter)
    return float(scores.mean())


def aipw_scores_cross_fitted(
    df: pd.DataFrame,
    outcome_col: str,
    treatment_col: str,
    covariate_cols: list,
    propensity_col: str,
    n_splits: int = 5,
    max_iter: int = 5000,
    random_state: int = None,
) -> np.ndarray:
    T = df[treatment_col].to_numpy()
    Y = df[outcome_col].to_numpy()
    X = df[covariate_cols].to_numpy()
    e = df[propensity_col].to_numpy()
    n = len(df)

    mu1 = np.empty(n)
    mu0 = np.empty(n)

    splitter = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state)

    for train_idx, holdout_idx in splitter.split(X, T):
        X_train = X[train_idx]
        T_train = T[train_idx]
        Y_train = Y[train_idx]
        X_hold = X[holdout_idx]

        model_treated = _outcome_model(max_iter)
        model_treated.fit(X_train[T_train == 1], Y_train[T_train == 1])
        mu1[holdout_idx] = model_treated.predict_proba(X_hold)[:, 1]

        model_control = _outcome_model(max_iter)
        model_control.fit(X_train[T_train == 0], Y_train[T_train == 0])
        mu0[holdout_idx] = model_control.predict_proba(X_hold)[:, 1]

    aipw_treated = mu1 + T * (Y - mu1) / e
    aipw_control = mu0 + (1 - T) * (Y - mu0) / (1 - e)

    return aipw_treated - aipw_control


def aipw_ate_cross_fitted(
    df: pd.DataFrame,
    outcome_col: str,
    treatment_col: str,
    covariate_cols: list,
    propensity_col: str,
    n_splits: int = 5,
    max_iter: int = 5000,
    random_state: int = None,
) -> float:
    scores = aipw_scores_cross_fitted(
        df, outcome_col, treatment_col, covariate_cols, propensity_col,
        n_splits=n_splits, max_iter=max_iter, random_state=random_state,
    )
    return float(scores.mean())


def run_estimator_comparison(
    df: pd.DataFrame,
    outcome_col: str,
    treatment_col: str,
    covariate_cols: list,
    caliper: float = 0.2,
    trim_method: str = "overlap",
    cross_fit: bool = True,
    n_splits: int = 5,
    random_state: int = None,
) -> dict:
    with_ps = attach_propensity(
        df, treatment_col, covariate_cols,
        cross_fit=cross_fit, n_splits=n_splits, random_state=random_state,
    )
    trimmed = apply_common_support_trim(with_ps, PROPENSITY_COL, treatment_col, method=trim_method)

    if cross_fit:
        aipw_estimate = aipw_ate_cross_fitted(
            trimmed, outcome_col, treatment_col, covariate_cols, PROPENSITY_COL,
            n_splits=n_splits, random_state=random_state,
        )
    else:
        aipw_estimate = aipw_ate(trimmed, outcome_col, treatment_col, covariate_cols, PROPENSITY_COL)

    results = {
        "naive_ols": naive_ols_ate(trimmed, outcome_col, treatment_col),
        "psm": psm_ate(trimmed, outcome_col, treatment_col, PROPENSITY_COL, caliper=caliper, random_state=random_state),
        "ipw": ipw_ate(trimmed, outcome_col, treatment_col, PROPENSITY_COL),
        "aipw": aipw_estimate,
    }

    for method, estimate in results.items():
        logger.info("Estimator '%s': ATE = %.5f", method, estimate)

    return results


def run_estimator_comparison_with_ci(
    df: pd.DataFrame,
    outcome_col: str,
    treatment_col: str,
    covariate_cols: list,
    caliper: float = 0.2,
    trim_method: str = "overlap",
    n_bootstrap: int = 200,
    alpha: float = 0.05,
    cross_fit: bool = True,
    n_splits: int = 5,
    random_state: int = None,
) -> dict:
    with_ps = attach_propensity(
        df, treatment_col, covariate_cols,
        cross_fit=cross_fit, n_splits=n_splits, random_state=random_state,
    )
    trimmed = apply_common_support_trim(with_ps, PROPENSITY_COL, treatment_col, method=trim_method)

    results = {}

    results["naive_ols"] = bootstrap_ci(
        trimmed,
        lambda d: naive_ols_ate(d, outcome_col, treatment_col),
        n_bootstrap=n_bootstrap,
        alpha=alpha,
        random_state=random_state,
    )

    matched = get_matched_pairs(trimmed, treatment_col, PROPENSITY_COL, caliper=caliper, random_state=random_state)
    pair_diffs = psm_pair_diffs(matched, outcome_col, treatment_col)
    results["psm"] = bootstrap_mean_ci(pair_diffs, n_bootstrap=n_bootstrap, alpha=alpha, random_state=random_state)

    results["ipw"] = bootstrap_ci(
        trimmed,
        lambda d: ipw_ate(d, outcome_col, treatment_col, PROPENSITY_COL),
        n_bootstrap=n_bootstrap,
        alpha=alpha,
        random_state=random_state,
    )

    if cross_fit:
        aipw_score_arr = aipw_scores_cross_fitted(
            trimmed, outcome_col, treatment_col, covariate_cols, PROPENSITY_COL,
            n_splits=n_splits, random_state=random_state,
        )
    else:
        aipw_score_arr = aipw_scores(trimmed, outcome_col, treatment_col, covariate_cols, PROPENSITY_COL)
    results["aipw"] = bootstrap_mean_ci(aipw_score_arr, n_bootstrap=n_bootstrap, alpha=alpha, random_state=random_state)

    for method, r in results.items():
        logger.info(
            "Estimator '%s': ATE = %.5f [%.5f, %.5f]",
            method, r["point_estimate"], r["ci_lower"], r["ci_upper"],
        )

    return {
        method: {
            "point_estimate": r["point_estimate"],
            "ci_lower": r["ci_lower"],
            "ci_upper": r["ci_upper"],
        }
        for method, r in results.items()
    }
