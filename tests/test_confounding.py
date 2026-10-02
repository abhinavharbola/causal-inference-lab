import numpy as np
import pandas as pd
import pytest

from src.validation.confounding import (
    calibrate_confounding,
    check_confounding_validity,
    compute_retention_probabilities,
    induce_confounding,
    run_dose_response_confounding,
    select_confounding_covariate,
)


def _make_rct(seed: int, n: int = 60_000, treated_share: float = 0.85):
    rng = np.random.default_rng(seed)
    x = rng.normal(0, 1, n)
    noise = rng.normal(0, 1, n)
    treatment = rng.binomial(1, treated_share, n)
    base = 1 / (1 + np.exp(-(-2.5 + 0.8 * x)))
    visit = rng.binomial(1, np.clip(base + 0.03 * treatment, 0.001, 0.999))
    df = pd.DataFrame({"x": x, "noise": noise, "treatment": treatment, "visit": visit})
    truth = df.loc[df.treatment == 1, "visit"].mean() - df.loc[df.treatment == 0, "visit"].mean()
    return df, truth


@pytest.fixture
def rct():
    return _make_rct(0)


def test_select_confounding_covariate_picks_most_outcome_correlated_column(rct):
    df, _ = rct
    assert select_confounding_covariate(df, "visit", ["x", "noise"]) == "x"


def test_retention_probabilities_keep_marginal_retention_constant_across_x():
    x = np.linspace(-4, 4, 41)
    share = 0.85
    keep = 0.5
    for g2 in (0.0, 0.5, 1.5, 3.0):
        treated = compute_retention_probabilities(x, np.ones_like(x), g2, share, keep)
        control = compute_retention_probabilities(x, np.zeros_like(x), g2, share, keep)
        marginal = share * treated + (1 - share) * control
        assert np.allclose(marginal, keep)
        assert treated.min() >= 0 and treated.max() <= 1
        assert control.min() >= 0 and control.max() <= 1


def test_retention_probabilities_are_identical_across_arms_free_of_x_at_zero_g2():
    x = np.linspace(-3, 3, 13)
    treated = compute_retention_probabilities(x, np.ones_like(x), 0.0, 0.85, 0.5)
    control = compute_retention_probabilities(x, np.zeros_like(x), 0.0, 0.85, 0.5)
    assert np.allclose(treated, treated[0])
    assert np.allclose(control, control[0])


def test_retention_probabilities_reject_degenerate_shares():
    x = np.zeros(3)
    with pytest.raises(ValueError):
        compute_retention_probabilities(x, np.ones(3), 1.0, 1.0, 0.5)
    with pytest.raises(ValueError):
        compute_retention_probabilities(x, np.ones(3), 1.0, 0.85, 0.0)


def test_induce_confounding_preserves_the_covariate_marginal(rct):
    df, _ = rct
    retained = induce_confounding(df, "x", "treatment", g2=1.5, random_state=1)

    assert retained["x"].mean() == pytest.approx(df["x"].mean(), abs=0.03)
    assert retained["x"].std() == pytest.approx(df["x"].std(), abs=0.03)
    assert len(retained) / len(df) == pytest.approx(0.5, abs=0.02)


def test_induce_confounding_correlates_treatment_with_x_only_when_g2_positive(rct):
    df, _ = rct
    flat = induce_confounding(df, "x", "treatment", g2=0.0, random_state=1)
    tilted = induce_confounding(df, "x", "treatment", g2=1.5, random_state=1)

    assert abs(np.corrcoef(flat["x"], flat["treatment"])[0, 1]) < 0.02
    assert np.corrcoef(tilted["x"], tilted["treatment"])[0, 1] > 0.1


def test_induce_confounding_rejects_constant_covariate(rct):
    df, _ = rct
    df = df.assign(x=1.0)
    with pytest.raises(ValueError):
        induce_confounding(df, "x", "treatment", g2=1.0)


def test_naive_bias_grows_with_g2(rct):
    df, truth = rct
    biases = []
    for g2 in (0.0, 0.75, 1.5):
        retained = induce_confounding(df, "x", "treatment", g2=g2, random_state=2)
        diagnostics = check_confounding_validity(retained, "x", "treatment", "visit", truth)
        biases.append(diagnostics["naive_estimate"] - truth)

    assert biases[2] > biases[1] > biases[0]
    assert biases[2] > 0.01


def test_gate_passes_under_strong_confounding_across_seeds():
    passes = 0
    for seed in range(5):
        df, truth = _make_rct(seed)
        retained = induce_confounding(df, "x", "treatment", g2=1.5, random_state=seed)
        passes += check_confounding_validity(retained, "x", "treatment", "visit", truth)["passes_gate"]
    assert passes == 5


def test_gate_rarely_passes_without_confounding_across_seeds():
    false_passes = 0
    n_seeds = 20
    for seed in range(n_seeds):
        df, truth = _make_rct(seed)
        retained = induce_confounding(df, "x", "treatment", g2=0.0, random_state=seed)
        false_passes += check_confounding_validity(retained, "x", "treatment", "visit", truth)["passes_gate"]
    assert false_passes <= 3


def test_gate_compares_truth_with_the_retained_sample_interval(rct):
    df, truth = rct
    retained = induce_confounding(df, "x", "treatment", g2=0.0, random_state=3)
    diagnostics = check_confounding_validity(retained, "x", "treatment", "visit", truth)

    lower, upper = diagnostics["naive_ci"]
    assert lower < diagnostics["naive_estimate"] < upper
    assert (upper - lower) > 0.005


def test_calibrate_confounding_converges_with_a_relevant_covariate(rct):
    df, truth = rct
    result = calibrate_confounding(df, "x", "treatment", "visit", truth, random_state=0)

    assert result["converged"]
    assert result["g2"] > 0
    assert result["history"][-1]["passes_gate"]


def test_calibrate_confounding_rarely_converges_with_an_irrelevant_covariate():
    n_converged = 0
    n_trials = 20
    for seed in range(n_trials):
        df, truth = _make_rct(seed)
        result = calibrate_confounding(df, "noise", "treatment", "visit", truth, max_iters=1, random_state=seed)
        n_converged += result["converged"]
    assert n_converged <= 4


def test_calibrate_confounding_returns_retained_data_and_history_after_max_iters(rct):
    df, truth = rct
    result = calibrate_confounding(df, "noise", "treatment", "visit", truth, max_iters=1, g2_init=0.0, random_state=0)

    assert result["retained_df"] is not None
    assert len(result["history"]) == 1


def test_run_dose_response_confounding_returns_each_severity_with_increasing_bias(rct):
    df, truth = rct
    severities = {"none": 0.0, "mild": 0.5, "strong": 1.5}
    results = run_dose_response_confounding(
        df, "x", "treatment", "visit", truth, severities=severities, random_state=0
    )

    assert set(results.keys()) == set(severities.keys())
    naive = [results[k]["diagnostics"]["naive_estimate"] for k in ("none", "mild", "strong")]
    assert naive[0] < naive[1] < naive[2]
