"""
Tests for src/validation/confounding.py.

Covers: covariate selection by outcome correlation, the retention rule's
behavior at g2=0 vs g2>0, the two-part validation gate, and both outcomes
of the calibration loop (converges on a real relationship, explicitly
reports non-convergence on an outcome-irrelevant covariate rather than
looping forever or silently accepting a bad g2).
"""

import numpy as np
import pandas as pd
import pytest

from src.utils.bootstrap import analytic_ci_diff_in_proportions
from src.validation.confounding import (
    select_confounding_covariate,
    compute_retention_probability,
    induce_confounding,
    check_confounding_validity,
    calibrate_confounding,
    run_dose_response_confounding,
)


@pytest.fixture
def synthetic_data():
    """
    A dataset where f0 drives the outcome and is randomized against
    treatment (RCT), f_noise is unrelated to the outcome, and treatment
    has a known true effect. Used across most tests in this file.
    """
    rng = np.random.default_rng(0)
    n = 50_000
    treatment = rng.binomial(1, 0.85, n)
    f0 = rng.normal(0, 1, n)
    f_noise = rng.normal(0, 1, n)

    true_ate = 0.01
    base_rate = 0.045
    p = np.clip(base_rate + true_ate * treatment + 0.03 * f0, 0.001, 0.999)
    visit = rng.binomial(1, p)

    df = pd.DataFrame({"f0": f0, "f_noise": f_noise, "treatment": treatment, "visit": visit})

    ground_truth = analytic_ci_diff_in_proportions(
        df.loc[df.treatment == 1, "visit"].mean(),
        (df.treatment == 1).sum(),
        df.loc[df.treatment == 0, "visit"].mean(),
        (df.treatment == 0).sum(),
    )
    return df, ground_truth


def test_select_confounding_covariate_picks_outcome_correlated_column(synthetic_data):
    df, _ = synthetic_data
    selected = select_confounding_covariate(df, "visit", ["f0", "f_noise"])
    assert selected == "f0"


def test_retention_probability_at_zero_g2_does_not_depend_on_treatment():
    """At g2=0 the X*T interaction term vanishes, so retention probability
    should be identical for a given X regardless of T."""
    X = np.array([-1.0, 0.0, 1.0, 2.0])
    T1 = np.ones_like(X)
    T0 = np.zeros_like(X)

    prob_treated = compute_retention_probability(X, T1, g0=0.0, g1=0.5, g2=0.0)
    prob_control = compute_retention_probability(X, T0, g0=0.0, g1=0.5, g2=0.0)

    np.testing.assert_allclose(prob_treated, prob_control)


def test_induce_confounding_at_nonzero_g2_creates_xt_correlation(synthetic_data):
    df, _ = synthetic_data
    retained = induce_confounding(df, "f0", "treatment", g0=0.0, g1=0.0, g2=2.0, random_state=1)

    corr = retained["f0"].corr(retained["treatment"])
    assert abs(corr) > 0.05


def test_check_confounding_validity_gate_requires_both_conditions(synthetic_data):
    df, ground_truth = synthetic_data
    ci = (ground_truth["ci_lower"], ground_truth["ci_upper"])

    # g2=0: no induced confounding, gate should fail (naive estimate should
    # stay within the ground-truth CI, and X shouldn't correlate with T).
    unconfounded = induce_confounding(df, "f0", "treatment", g0=0.0, g1=0.0, g2=0.0, random_state=1)
    result_none = check_confounding_validity(
        unconfounded, "f0", "treatment", "visit", ground_truth["point_estimate"], ci
    )
    assert not result_none["passes_gate"]

    # g2=3: strong induced confounding, gate should pass.
    confounded = induce_confounding(df, "f0", "treatment", g0=0.0, g1=0.0, g2=3.0, random_state=1)
    result_strong = check_confounding_validity(
        confounded, "f0", "treatment", "visit", ground_truth["point_estimate"], ci
    )
    assert result_strong["passes_gate"]
    assert result_strong["corr_significant"]
    assert result_strong["estimate_outside_ci"]


def test_calibrate_confounding_converges_on_outcome_relevant_covariate(synthetic_data):
    df, ground_truth = synthetic_data
    ci = (ground_truth["ci_lower"], ground_truth["ci_upper"])

    result = calibrate_confounding(
        df, "f0", "treatment", "visit", ground_truth["point_estimate"], ci,
        g2_init=0.5, g2_step=0.5, max_iters=10, random_state=1,
    )

    assert result["converged"]
    assert result["retained_df"] is not None
    assert len(result["history"]) > 0


def test_calibrate_confounding_reports_non_convergence_on_irrelevant_covariate(synthetic_data):
    """
    f_noise doesn't correlate with the outcome, so biasing retention by
    f_noise*T should almost never push the naive estimate outside the
    ground-truth CI. The validation gate uses alpha=0.05 for the
    correlation-significance check, so it has an inherent ~5%
    false-positive rate by construction (this is expected hypothesis-test
    behavior, not a bug) — a single seed can occasionally converge by
    chance. This test checks the behavior holds across several seeds
    rather than asserting a single draw, and confirms the loop always
    reports its convergence status explicitly (never hangs, never
    returns a None retained_df) regardless of outcome.
    """
    df, ground_truth = synthetic_data
    ci = (ground_truth["ci_lower"], ground_truth["ci_upper"])

    n_converged = 0
    n_trials = 10
    for seed in range(n_trials):
        result = calibrate_confounding(
            df, "f_noise", "treatment", "visit", ground_truth["point_estimate"], ci,
            g2_init=0.5, g2_step=0.5, max_iters=3, random_state=seed + 500,
        )
        assert result["retained_df"] is not None
        assert len(result["history"]) > 0
        if result["converged"]:
            n_converged += 1

    # Should rarely converge (expected false-positive rate ~alpha=0.05);
    # allow generous headroom since this is a small, noisy sample of seeds.
    assert n_converged <= 2, f"{n_converged}/{n_trials} seeds falsely converged on an outcome-irrelevant covariate"


def test_dose_response_confounding_bias_grows_with_severity(synthetic_data):
    df, ground_truth = synthetic_data
    ci = (ground_truth["ci_lower"], ground_truth["ci_upper"])

    severities = {"none": 0.0, "mild": 1.0, "strong": 3.0}
    result = run_dose_response_confounding(
        df, "f0", "treatment", "visit", ground_truth["point_estimate"], ci,
        severities=severities, random_state=1,
    )

    bias_none = abs(result["none"]["diagnostics"]["naive_estimate"] - ground_truth["point_estimate"])
    bias_strong = abs(result["strong"]["diagnostics"]["naive_estimate"] - ground_truth["point_estimate"])

    assert bias_strong > bias_none
    assert not result["none"]["diagnostics"]["passes_gate"]
    assert result["strong"]["diagnostics"]["passes_gate"]