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