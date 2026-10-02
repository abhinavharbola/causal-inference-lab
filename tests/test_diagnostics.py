import numpy as np
import pandas as pd
import pytest

from src.validation.diagnostics import (
    OVERLAP_COEFFICIENT_WARN,
    overlap_coefficient,
    overlap_diagnostics,
    run_full_diagnostics,
)
from src.validation.estimators import attach_propensity, fit_propensity_score, psm_pair_diffs, run_estimator_comparison


@pytest.fixture
def confounded_data():
    rng = np.random.default_rng(4)
    n = 20_000
    f0 = rng.normal(0, 1, n)
    f1 = rng.normal(0, 1, n)
    treat_prob = 1 / (1 + np.exp(-1.5 * f0))
    treatment = rng.binomial(1, treat_prob)
    visit = rng.binomial(1, np.clip(0.045 + 0.01 * treatment + 0.02 * f0, 0.001, 0.999))
    df = pd.DataFrame({"f0": f0, "f1": f1, "treatment": treatment, "visit": visit})
    df["_propensity"] = fit_propensity_score(df, "treatment", ["f0", "f1"])
    return df


@pytest.fixture
def bucketed_confounded_data():
    rng = np.random.default_rng(5)
    n = 20_000
    f0 = rng.choice(np.arange(6, dtype=float), size=n)
    treat_prob = 1 / (1 + np.exp(-0.4 * f0))
    treatment = rng.binomial(1, treat_prob)
    visit = rng.binomial(1, np.clip(0.045 + 0.01 * treatment, 0.001, 0.999))
    df = pd.DataFrame({"f0": f0, "treatment": treatment, "visit": visit})
    df["_propensity"] = fit_propensity_score(df, "treatment", ["f0"])
    return df


def test_run_full_diagnostics_reports_consistent_match_counts(confounded_data):
    result = run_full_diagnostics(
        confounded_data, ["f0", "f1"], "treatment", "_propensity", caliper=0.2, random_state=0
    )

    assert result["n_pairs"] <= result["n_treated_total"]
    assert 0.0 <= result["match_rate"] <= 1.0
    assert result["n_control_unique"] == result["n_pairs"]


def test_run_full_diagnostics_match_rate_reflects_minority_group_size(confounded_data):
    n_control_total = (confounded_data["treatment"] == 0).sum()

    result = run_full_diagnostics(
        confounded_data, ["f0", "f1"], "treatment", "_propensity", caliper=0.2, random_state=0
    )

    assert result["n_pairs"] <= n_control_total
    assert result["match_rate"] == pytest.approx(result["n_pairs"] / result["n_treated_total"])


def test_run_full_diagnostics_measures_overlap_on_the_untrimmed_population(confounded_data):
    result = run_full_diagnostics(
        confounded_data, ["f0", "f1"], "treatment", "_propensity", caliper=0.2, random_state=0
    )

    assert result["overlap"]["n_total"] == len(confounded_data)


def test_run_full_diagnostics_matches_within_the_trimmed_pool(confounded_data):
    narrowed = confounded_data.copy()
    narrowed.loc[narrowed.index[:5], "_propensity"] = 0.001
    narrowed.loc[narrowed.index[:5], "treatment"] = 1

    result = run_full_diagnostics(
        narrowed, ["f0", "f1"], "treatment", "_propensity", caliper=0.2, trim_method="fixed", random_state=0
    )

    assert result["matched_df"]["_propensity"].between(0.1, 0.9).all()


def test_run_full_diagnostics_exposes_overlap_coefficient(confounded_data):
    result = run_full_diagnostics(
        confounded_data, ["f0", "f1"], "treatment", "_propensity", caliper=0.2, random_state=0
    )

    assert 0.0 <= result["overlap_coefficient"] <= 1.0
    assert result["overlap_coefficient"] == result["overlap"]["overlap_coefficient"]


def test_overlap_coefficient_is_one_for_identical_distributions():
    rng = np.random.default_rng(0)
    sample = rng.beta(2, 5, 50_000)
    assert overlap_coefficient(sample, sample.copy()) == pytest.approx(1.0, abs=1e-9)


def test_overlap_coefficient_is_zero_for_disjoint_distributions():
    treated = np.linspace(0.8, 0.95, 1_000)
    control = np.linspace(0.05, 0.2, 1_000)
    assert overlap_coefficient(treated, control) == pytest.approx(0.0, abs=1e-9)


def test_overlap_coefficient_catches_weak_overlap_that_the_range_check_misses():
    rng = np.random.default_rng(1)
    n = 20_000
    treatment = np.r_[np.ones(n // 2), np.zeros(n // 2)].astype(int)
    propensity = np.where(treatment == 1, 0.9, 0.1) + rng.normal(0, 0.01, n)
    propensity[:20] = 0.05
    propensity[-20:] = 0.95
    weak = pd.DataFrame({"treatment": treatment, "_propensity": np.clip(propensity, 0.001, 0.999), "visit": 0})

    result = overlap_diagnostics(weak, "_propensity", "treatment")

    assert result["pct_within_overlap"] > 95
    assert result["overlap_coefficient"] < OVERLAP_COEFFICIENT_WARN


def test_range_check_reports_no_overlap_for_fully_disjoint_arms():
    treatment = np.r_[np.ones(100), np.zeros(100)].astype(int)
    propensity = np.r_[np.linspace(0.8, 0.95, 100), np.linspace(0.05, 0.2, 100)]
    disjoint = pd.DataFrame({"treatment": treatment, "_propensity": propensity})

    result = overlap_diagnostics(disjoint, "_propensity", "treatment")

    assert result["pct_within_overlap"] == 0.0
    assert result["overlap_coefficient"] == pytest.approx(0.0, abs=1e-9)


def test_run_full_diagnostics_counts_control_rows_not_covariate_values(bucketed_confounded_data):
    result = run_full_diagnostics(
        bucketed_confounded_data, ["f0"], "treatment", "_propensity", caliper=0.2, random_state=0
    )
    matched_controls = result["matched_df"].loc[result["matched_df"]["treatment"] == 0]

    assert result["n_pairs"] > 0
    assert matched_controls["f0"].duplicated().any()
    assert matched_controls["f0"].nunique() < len(matched_controls)
    assert result["n_control_unique"] == result["n_pairs"]


def test_run_full_diagnostics_reproduces_the_pairs_used_by_the_psm_estimator():
    rng = np.random.default_rng(8)
    n = 20_000
    f0 = rng.normal(0, 1, n)
    f1 = rng.normal(0, 1, n)
    treatment = rng.binomial(1, 1 / (1 + np.exp(-(1.2 + 0.8 * f0))))
    visit = rng.binomial(1, np.clip(0.05 + 0.02 * treatment + 0.02 * f0, 0.001, 0.999))
    df = pd.DataFrame({"f0": f0, "f1": f1, "treatment": treatment, "visit": visit})

    with_ps = attach_propensity(df, "treatment", ["f0", "f1"], cross_fit=True, random_state=0)
    diagnostics = run_full_diagnostics(with_ps, ["f0", "f1"], "treatment", "_propensity", random_state=0)
    estimator_psm = run_estimator_comparison(df, "visit", "treatment", ["f0", "f1"], random_state=0)["psm"]

    pair_diffs = psm_pair_diffs(diagnostics["matched_df"], "visit", "treatment")

    assert pair_diffs.mean() == pytest.approx(estimator_psm, abs=1e-12)
