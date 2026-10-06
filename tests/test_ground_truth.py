import numpy as np
import pandas as pd

from src.validation.ground_truth import adjusted_ground_truth, standardized_mean_differences


def _confounded_data(n: int, seed: int) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    x0 = rng.normal(size=n)
    x1 = rng.normal(size=n)
    propensity = 1 / (1 + np.exp(-(1.0 + 1.2 * x0)))
    treatment = rng.binomial(1, propensity)
    outcome_prob = 0.04 + 0.02 * treatment + 0.08 / (1 + np.exp(-1.5 * x0))
    outcome = rng.binomial(1, outcome_prob)
    return pd.DataFrame({"x0": x0, "x1": x1, "treatment": treatment, "visit": outcome})


def test_adjusted_ground_truth_removes_confounding_that_the_raw_difference_keeps():
    df = _confounded_data(80_000, seed=0)

    result = adjusted_ground_truth(df, "treatment", ["visit"], ["x0", "x1"], n_splits=3, random_state=0)
    visit = result["outcomes"]["visit"]

    assert abs(visit["point_estimate"] - 0.02) < 4 * visit["se"]
    assert visit["naive_estimate"] - 0.02 > 0.01
    assert visit["ci_lower"] < visit["point_estimate"] < visit["ci_upper"]


def test_adjusted_ground_truth_agrees_with_the_raw_difference_under_randomization():
    rng = np.random.default_rng(1)
    n = 80_000
    df = pd.DataFrame(
        {
            "x0": rng.normal(size=n),
            "x1": rng.normal(size=n),
            "treatment": rng.binomial(1, 0.85, n),
        }
    )
    df["visit"] = rng.binomial(1, 0.04 + 0.01 * df["treatment"] + 0.02 / (1 + np.exp(-df["x0"])))

    result = adjusted_ground_truth(df, "treatment", ["visit"], ["x0", "x1"], n_splits=3, random_state=0)
    visit = result["outcomes"]["visit"]

    assert abs(visit["point_estimate"] - visit["naive_estimate"]) < 0.004
    assert result["propensity_auc"] < 0.53


def test_adjusted_ground_truth_returns_every_requested_outcome():
    df = _confounded_data(20_000, seed=2)
    df["conversion"] = (df["visit"] & (np.random.default_rng(2).uniform(size=len(df)) < 0.3)).astype(int)

    result = adjusted_ground_truth(df, "treatment", ["visit", "conversion"], ["x0", "x1"], n_splits=2, random_state=0)

    assert set(result["outcomes"].keys()) == {"visit", "conversion"}
    assert result["n"] == 20_000


def test_standardized_mean_differences_flags_the_shifted_covariate():
    df = _confounded_data(40_000, seed=3)

    smd = standardized_mean_differences(df, "treatment", ["x0", "x1"])

    assert smd["x0"] > 0.3
    assert abs(smd["x1"]) < 0.05
