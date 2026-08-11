import logging

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sortedcontainers import SortedList
import statsmodels.api as sm

from src.utils.bootstrap import bootstrap_ci, bootstrap_mean_ci

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
    """Per-pair (treated outcome - control outcome), for pair-level bootstrapping.

    Resampling whole pairs (rather than individual rows) is what preserves the
    matched design when computing a bootstrap CI for `psm_ate`.
    """
    treated = matched_df.loc[matched_df[treatment_col] == 1, [pair_id_col, outcome_col]]
    control = matched_df.loc[matched_df[treatment_col] == 0, [pair_id_col, outcome_col]]
    merged = treated.merge(control, on=pair_id_col, suffixes=("_treated", "_control"))
    return (merged[f"{outcome_col}_treated"] - merged[f"{outcome_col}_control"]).to_numpy()


def get_matched_pairs(
    df: pd.DataFrame,
    treatment_col: str,
    propensity_col: str,
    caliper: float = 0.2,
    random_state: int = None,
) -> pd.DataFrame:
    """Greedy 1:1 nearest-neighbor matching WITHOUT replacement.

    Each control unit is matched to at most one treated unit. This matters for two
    reasons downstream: (1) Rosenbaum sensitivity bounds and any pair-level variance
    estimate assume independent matched pairs, which breaks if the same control is
    reused across many pairs; (2) a matched-pairs balance/overlap diagnostic is only
    meaningful if "pairs" really are 1:1 correspondences.

    Because control units cannot be reused, and this dataset has an imbalanced
    treatment allocation (~85% treated), the number of matched pairs is capped by
    the size of the control pool: most treated units will go unmatched even before
    the caliper is applied. That is expected, correct behavior for strict 1:1
    matching under an imbalanced design, not a bug — see `match_rate` in
    `run_full_diagnostics` for how much of the treated group this discards.

    Matching order is randomized (not sorted by propensity) so results don't
    systematically favor either tail of the propensity distribution when the
    control pool runs out; pass `random_state` for reproducibility.
    """
    treated = df[df[treatment_col] == 1].reset_index(drop=True)
    control = df[df[treatment_col] == 0].reset_index(drop=True)

    ps_std = df[propensity_col].std()
    caliper_distance = caliper * ps_std

    rng = np.random.default_rng(random_state)
    match_order = rng.permutation(len(treated))

    # SortedList of (propensity, control_row_index) gives O(log n) nearest-neighbor
    # lookup AND O(log n) removal, so strict without-replacement matching stays
    # tractable at this project's scale (tens of thousands of controls).
    available = SortedList((control.loc[i, propensity_col], i) for i in control.index)

    matched_treated_idx = []
    matched_control_idx = []

    for t_idx in match_order:
        if len(available) == 0:
            break

        t_ps = treated.loc[t_idx, propensity_col]
        pos = available.bisect_left((t_ps, -1))

        candidates = []
        if pos < len(available):
            candidates.append(available[pos])
        if pos > 0:
            candidates.append(available[pos - 1])

        best_ps, best_idx = min(candidates, key=lambda c: abs(c[0] - t_ps))
        if abs(best_ps - t_ps) <= caliper_distance:
            matched_treated_idx.append(t_idx)
            matched_control_idx.append(best_idx)
            available.remove((best_ps, best_idx))

    n_unmatched = len(treated) - len(matched_treated_idx)
    if n_unmatched > 0:
        logger.info(
            "PSM (1:1, no replacement): %d of %d treated units unmatched "
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


def aipw_scores(
    df: pd.DataFrame,
    outcome_col: str,
    treatment_col: str,
    covariate_cols: list,
    propensity_col: str,
    max_iter: int = 5000,
) -> np.ndarray:
    """Per-unit doubly-robust pseudo-outcome (aipw_treated - aipw_control).

    `aipw_ate` is just the mean of this array. Exposed separately so the outcome
    models can be fit ONCE and the resulting fixed scores bootstrapped cheaply
    (see `run_estimator_comparison_with_ci`), instead of refitting two logistic
    regressions on every bootstrap resample.
    """
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


def run_estimator_comparison(
    df: pd.DataFrame,
    outcome_col: str,
    treatment_col: str,
    covariate_cols: list,
    caliper: float = 0.2,
    trim_method: str = "overlap",
    random_state: int = None,
) -> dict:
    """Point estimates only (fast). Use `run_estimator_comparison_with_ci` for CIs."""
    propensity = fit_propensity_score(df, treatment_col, covariate_cols)
    df = df.copy()
    df["_propensity"] = propensity

    trimmed = apply_common_support_trim(df, "_propensity", treatment_col, method=trim_method)

    results = {
        "naive_ols": naive_ols_ate(df, outcome_col, treatment_col),
        "psm": psm_ate(trimmed, outcome_col, treatment_col, "_propensity", caliper=caliper, random_state=random_state),
        "ipw": ipw_ate(trimmed, outcome_col, treatment_col, "_propensity"),
        "aipw": aipw_ate(trimmed, outcome_col, treatment_col, covariate_cols, "_propensity"),
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
    random_state: int = None,
) -> dict:
    """Point estimate + real bootstrap CI for all four estimators.

    Each method is bootstrapped in whatever way is both valid and cheap for it,
    rather than refitting every model from scratch on every resample:
      - naive_ols: resample rows, refit OLS each time (fast, no ML model).
      - psm: resample whole matched PAIRS (not raw rows), preserving the paired
        design that the point estimate itself relies on.
      - ipw: resample rows, recompute the Hajek ratio using the already-fitted
        propensity scores (cheap arithmetic, no refitting).
      - aipw: fit the two outcome-regression models ONCE, reduce to a fixed
        per-unit score, then bootstrap that array. Refitting AIPW's nuisance
        models on every one of n_bootstrap resamples would be far too slow to
        run repeatedly on a CPU-only machine.

    Returns {method: {point_estimate, ci_lower, ci_upper}}.
    """
    propensity = fit_propensity_score(df, treatment_col, covariate_cols)
    df = df.copy()
    df["_propensity"] = propensity
    trimmed = apply_common_support_trim(df, "_propensity", treatment_col, method=trim_method)

    results = {}

    naive_boot = bootstrap_ci(
        df,
        lambda d: naive_ols_ate(d, outcome_col, treatment_col),
        n_bootstrap=n_bootstrap,
        alpha=alpha,
        random_state=random_state,
    )
    results["naive_ols"] = naive_boot

    matched = get_matched_pairs(trimmed, treatment_col, "_propensity", caliper=caliper, random_state=random_state)
    pair_diffs = psm_pair_diffs(matched, outcome_col, treatment_col)
    results["psm"] = bootstrap_mean_ci(pair_diffs, n_bootstrap=n_bootstrap, alpha=alpha, random_state=random_state)

    ipw_boot = bootstrap_ci(
        trimmed,
        lambda d: ipw_ate(d, outcome_col, treatment_col, "_propensity"),
        n_bootstrap=n_bootstrap,
        alpha=alpha,
        random_state=random_state,
    )
    results["ipw"] = ipw_boot

    aipw_score_arr = aipw_scores(trimmed, outcome_col, treatment_col, covariate_cols, "_propensity")
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