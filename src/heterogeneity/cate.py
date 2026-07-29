"""
Section 2: T-learner CATE estimation on the full untouched randomized
Criteo data (visit as outcome).

T-learner is used as the primary method rather than a causal forest:
simpler, more interpretable, and easier to defend and explain than a
causal forest in an interview setting. Causal forest / X-learner is
documented as a natural extension in the README, not built here, to
protect feasibility.

Base classifiers are calibrated (CalibratedClassifierCV) rather than
used raw. Visit has a low base rate (~4-5%), and uncalibrated
probability estimates from tree-based or even logistic models can be
systematically off at that base rate, which then directly noises up the
CATE difference (mu1_hat - mu0_hat), since CATE is a difference of two
already-imperfect probability estimates.

The default base estimator standardizes features before fitting logistic
regression. Development/testing used standard-normal synthetic features,
which converge under lbfgs regardless of scaling, so this wasn't caught
until a real run against actual Criteo data surfaced both an lbfgs
non-convergence warning and an implausibly wide CATE range (consistent
with unscaled features destabilizing both the logistic fit and the
isotonic calibration in sparse regions of an unscaled feature space).
Calibration defaults to 'sigmoid' (Platt scaling) rather than 'isotonic':
sigmoid is a simple 2-parameter fit and much less prone to producing
erratic extreme values in sparse folds, which matters here since the
control arm is meaningfully smaller than the treated arm (~15% of the
sample at Criteo's actual treatment share) and gets split further by
cross-validation during calibration.
"""

import logging

import numpy as np
import pandas as pd
from sklearn.calibration import CalibratedClassifierCV
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

logger = logging.getLogger(__name__)


def _default_base_estimator():
    return make_pipeline(StandardScaler(), LogisticRegression(max_iter=5000))


def fit_t_learner(
    train_df: pd.DataFrame,
    feature_cols: list,
    treatment_col: str,
    outcome_col: str,
    base_estimator=None,
    calibrate: bool = True,
    calibration_method: str = "sigmoid",
    cv: int = 3,
) -> dict:
    """
    Fits two separate outcome models, one on treated units and one on
    control units, each predicting P(outcome=1 | X). CATE is the
    difference in their predictions at inference time.

    base_estimator defaults to a StandardScaler + LogisticRegression
    pipeline, a reasonable and cheap default given the CPU-only
    constraint; swap in a tree-based model if the covariates show clear
    nonlinearity, but scale and recalibrate either way. If you pass a
    custom base_estimator, this function does not add scaling for you —
    build scaling into your own pipeline if your estimator needs it.
    """
    if base_estimator is None:
        base_estimator = _default_base_estimator()

    treated = train_df[train_df[treatment_col] == 1]
    control = train_df[train_df[treatment_col] == 0]

    X_treated = treated[feature_cols].to_numpy()
    y_treated = treated[outcome_col].to_numpy()
    X_control = control[feature_cols].to_numpy()
    y_control = control[outcome_col].to_numpy()

    if calibrate:
        model_treated = CalibratedClassifierCV(base_estimator, cv=cv, method=calibration_method)
        model_control = CalibratedClassifierCV(base_estimator, cv=cv, method=calibration_method)
    else:
        from sklearn.base import clone
        model_treated = clone(base_estimator)
        model_control = clone(base_estimator)

    model_treated.fit(X_treated, y_treated)
    model_control.fit(X_control, y_control)

    logger.info(
        "T-learner fit: %d treated units, %d control units, calibrate=%s (method=%s)",
        len(treated),
        len(control),
        calibrate,
        calibration_method if calibrate else "n/a",
    )

    return {
        "model_treated": model_treated,
        "model_control": model_control,
        "feature_cols": feature_cols,
    }


def predict_cate(models: dict, df: pd.DataFrame) -> np.ndarray:
    """
    Predicts CATE = P(outcome=1 | X, T=1) - P(outcome=1 | X, T=0) for
    every row in df, regardless of that row's actual treatment status.
    This is the whole point of the T-learner: predict both potential
    outcomes for every unit.
    """
    X = df[models["feature_cols"]].to_numpy()

    p1 = models["model_treated"].predict_proba(X)[:, 1]
    p0 = models["model_control"].predict_proba(X)[:, 1]

    return p1 - p0


def t_learner_pipeline(
    df: pd.DataFrame,
    feature_cols: list,
    treatment_col: str,
    outcome_col: str,
    base_estimator=None,
    calibrate: bool = True,
    calibration_method: str = "sigmoid",
    cv: int = 3,
    test_size: float = 0.3,
    random_state: int = None,
) -> dict:
    """
    Full Section 2 CATE pipeline: splits into train/holdout (stratified
    by treatment so both arms are represented in each split), fits the
    T-learner on train, predicts CATE on the holdout set, and returns
    everything downstream evaluation (Qini/uplift curve) and segmentation
    need: the holdout dataframe with a `cate` column attached, plus the
    fitted models for reuse.
    """
    train_df, holdout_df = train_test_split(
        df,
        test_size=test_size,
        stratify=df[treatment_col],
        random_state=random_state,
    )

    models = fit_t_learner(
        train_df, feature_cols, treatment_col, outcome_col,
        base_estimator=base_estimator, calibrate=calibrate, calibration_method=calibration_method, cv=cv,
    )

    holdout_df = holdout_df.copy()
    holdout_df["cate"] = predict_cate(models, holdout_df)

    logger.info(
        "Holdout CATE summary: mean=%.5f, std=%.5f, min=%.5f, max=%.5f",
        holdout_df["cate"].mean(),
        holdout_df["cate"].std(),
        holdout_df["cate"].min(),
        holdout_df["cate"].max(),
    )

    return {
        "models": models,
        "train_df": train_df,
        "holdout_df": holdout_df,
    }