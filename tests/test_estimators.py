import numpy as np
import pandas as pd
import pytest

from src.validation.estimators import (
    naive_ols_ate,
    fit_propensity_score,
    apply_common_support_trim,
    get_matched_pairs,
    psm_ate,
    psm_pair_diffs,
    ipw_ate,
    aipw_ate,
    aipw_scores,
    run_estimator_comparison,
    run_estimator_comparison_with_ci,
)


@pytest.fixture
def rct_data():
    rng = np.random.default_rng(0)
    n = 50_000
    treatment = rng.binomial(1, 0.85, n)
    f0 = rng.normal(0, 1, n)
    f1 = rng.normal(0, 1, n)

    true_ate = 0.01
    base_rate = 0.045
    p = np.clip(base_rate + true_ate * treatment + 0.03 * f0, 0.001, 0.999)
    visit = rng.binomial(1, p)

    df = pd.DataFrame({"f0": f0, "f1": f1, "treatment": treatment, "visit": visit})
    return df, true_ate


def test_naive_ols_matches_diff_in_means_without_covariates(rct_data):
    df, _ = rct_data
    ols_estimate = naive_ols_ate(df, "visit", "treatment")

    treated_mean = df.loc[df.treatment == 1, "visit"].mean()
    control_mean = df.loc[df.treatment == 0, "visit"].mean()
    diff_in_means = treated_mean - control_mean

    assert ols_estimate == pytest.approx(diff_in_means, abs=1e-9)


def test_fit_propensity_score_is_clipped_away_from_zero_and_one(rct_data):
    df, _ = rct_data
    propensity = fit_propensity_score(df, "treatment", ["f0", "f1"])

    assert propensity.min() >= 1e-3
    assert propensity.max() <= 1 - 1e-3


def test_common_support_trim_overlap_keeps_only_shared_region(rct_data):
    df, _ = rct_data
    df = df.copy()
    df["_propensity"] = fit_propensity_score(df, "treatment", ["f0", "f1"])

    trimmed = apply_common_support_trim(df, "_propensity", "treatment", method="overlap")

    treated_ps = df.loc[df.treatment == 1, "_propensity"]
    control_ps = df.loc[df.treatment == 0, "_propensity"]
    expected_lower = max(treated_ps.min(), control_ps.min())
    expected_upper = min(treated_ps.max(), control_ps.max())

    assert trimmed["_propensity"].min() >= expected_lower
    assert trimmed["_propensity"].max() <= expected_upper
    assert len(trimmed) <= len(df)


def test_common_support_trim_fixed_band_respects_bounds(rct_data):
    df, _ = rct_data
    df = df.copy()
    df["_propensity"] = fit_propensity_score(df, "treatment", ["f0", "f1"])

    band = (df["_propensity"].quantile(0.05), df["_propensity"].quantile(0.95))
    trimmed = apply_common_support_trim(df, "_propensity", "treatment", method="fixed", fixed_bounds=band)

    assert len(trimmed) > 0
    assert trimmed["_propensity"].min() >= band[0]
    assert trimmed["_propensity"].max() <= band[1]


def test_get_matched_pairs_produces_equal_treated_and_control_counts(rct_data):
    df, _ = rct_data
    df = df.copy()
    df["_propensity"] = fit_propensity_score(df, "treatment", ["f0", "f1"])

    matched = get_matched_pairs(df, "treatment", "_propensity", caliper=0.2)

    n_treated = (matched["treatment"] == 1).sum()
    n_control = (matched["treatment"] == 0).sum()
    assert n_treated == n_control  # one control matched per treated unit
    assert matched["_pair_id"].nunique() == n_treated


def test_get_matched_pairs_improves_covariate_balance(rct_data):
    rng = np.random.default_rng(2)
    n = 20_000
    f0 = rng.normal(0, 1, n)
    # Treatment probability strongly depends on f0 (confounded, not RCT).
    treat_prob = 1 / (1 + np.exp(-2 * f0))
    treatment = rng.binomial(1, treat_prob)
    visit = rng.binomial(1, np.clip(0.045 + 0.01 * treatment, 0.001, 0.999))
    df = pd.DataFrame({"f0": f0, "treatment": treatment, "visit": visit})
    df["_propensity"] = fit_propensity_score(df, "treatment", ["f0"])

    matched = get_matched_pairs(df, "treatment", "_propensity", caliper=0.2)

    treated_f0 = matched.loc[matched.treatment == 1, "f0"]
    control_f0 = matched.loc[matched.treatment == 0, "f0"]

    assert abs(df.loc[df.treatment == 1, "f0"].mean() - df.loc[df.treatment == 0, "f0"].mean()) > 0.5
    assert abs(treated_f0.mean() - control_f0.mean()) < 0.05


def test_all_estimators_recover_approximately_true_ate_on_rct_data(rct_data):
    df, true_ate = rct_data
    results = run_estimator_comparison(df, "visit", "treatment", ["f0", "f1"])

    for method, estimate in results.items():
        assert estimate == pytest.approx(true_ate, abs=0.01), f"{method} estimate {estimate} too far from {true_ate}"


def test_ipw_ate_uses_stabilized_normalization_not_raw_group_count():
    rng = np.random.default_rng(3)
    n = 10_000
    treatment = rng.binomial(1, 0.85, n)
    visit = rng.binomial(1, np.clip(0.045 + 0.01 * treatment, 0.001, 0.999))
    propensity = np.clip(rng.beta(2, 8, n), 1e-3, 1 - 1e-3)

    df = pd.DataFrame({"treatment": treatment, "visit": visit, "_propensity": propensity})

    estimate = ipw_ate(df, "visit", "treatment", "_propensity")

    assert -0.2 < estimate < 0.2


def test_aipw_runs_and_returns_a_finite_scalar(rct_data):
    df, _ = rct_data
    df = df.copy()
    df["_propensity"] = fit_propensity_score(df, "treatment", ["f0", "f1"])

    estimate = aipw_ate(df, "visit", "treatment", ["f0", "f1"], "_propensity")

    assert np.isfinite(estimate)


def test_run_estimator_comparison_returns_all_four_methods(rct_data):
    df, _ = rct_data
    results = run_estimator_comparison(df, "visit", "treatment", ["f0", "f1"])

    assert set(results.keys()) == {"naive_ols", "psm", "ipw", "aipw"}
    for estimate in results.values():
        assert np.isfinite(estimate)


def test_get_matched_pairs_uses_each_control_at_most_once(rct_data):
    # Regression test for a bug where NearestNeighbors matching reused the same
    # control row across many treated units (matching with replacement), which
    # breaks the independence assumption Rosenbaum bounds rely on. Matching must
    # be strictly 1:1: every control row appears in at most one pair.
    #
    # Uses an explicit row-id column rather than dropping columns and
    # deduplicating on covariate values: covariate-based dedup silently
    # undercounts whenever two distinct rows share covariate values (e.g. the
    # real dataset's bucketed f-columns), which would mask exactly the bug
    # this test exists to catch. See test_diagnostics.py for the same fix
    # applied to run_full_diagnostics.
    df, _ = rct_data
    df = df.copy()
    df["_propensity"] = fit_propensity_score(df, "treatment", ["f0", "f1"])
    df["_row_id"] = np.arange(len(df))

    matched = get_matched_pairs(df, "treatment", "_propensity", caliper=0.2, random_state=0)

    n_pairs = matched["_pair_id"].nunique()
    n_unique_controls = matched.loc[matched.treatment == 0, "_row_id"].nunique()

    assert n_unique_controls == n_pairs


def test_get_matched_pairs_caps_matches_at_size_of_minority_group(rct_data):
    # With ~85% treated / ~15% control, strict 1:1 matching without replacement
    # cannot produce more matched pairs than there are control units available,
    # even before the caliper is applied.
    df, _ = rct_data
    df = df.copy()
    df["_propensity"] = fit_propensity_score(df, "treatment", ["f0", "f1"])
    n_control = (df["treatment"] == 0).sum()

    matched = get_matched_pairs(df, "treatment", "_propensity", caliper=0.2, random_state=0)
    n_pairs = matched["_pair_id"].nunique()

    assert n_pairs <= n_control


def test_psm_pair_diffs_mean_matches_psm_ate(rct_data):
    df, _ = rct_data
    df = df.copy()
    df["_propensity"] = fit_propensity_score(df, "treatment", ["f0", "f1"])

    matched = get_matched_pairs(df, "treatment", "_propensity", caliper=0.2, random_state=1)
    diffs = psm_pair_diffs(matched, "visit", "treatment")
    direct_estimate = psm_ate(df, "visit", "treatment", "_propensity", caliper=0.2, random_state=1)

    assert diffs.mean() == pytest.approx(direct_estimate, abs=1e-9)


def test_aipw_scores_mean_matches_aipw_ate(rct_data):
    df, _ = rct_data
    df = df.copy()
    df["_propensity"] = fit_propensity_score(df, "treatment", ["f0", "f1"])

    scores = aipw_scores(df, "visit", "treatment", ["f0", "f1"], "_propensity")
    estimate = aipw_ate(df, "visit", "treatment", ["f0", "f1"], "_propensity")

    assert scores.mean() == pytest.approx(estimate, abs=1e-9)


def test_run_estimator_comparison_with_ci_produces_valid_intervals(rct_data):
    df, true_ate = rct_data
    results = run_estimator_comparison_with_ci(
        df, "visit", "treatment", ["f0", "f1"], n_bootstrap=50, random_state=0
    )

    assert set(results.keys()) == {"naive_ols", "psm", "ipw", "aipw"}
    for method, r in results.items():
        assert r["ci_lower"] <= r["point_estimate"] <= r["ci_upper"], method
        # A real CI shouldn't collapse to a single fabricated width formula for
        # every method; at minimum it should have positive width.
        assert r["ci_upper"] > r["ci_lower"], method

