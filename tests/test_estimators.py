import numpy as np
import pandas as pd
import pytest
from sklearn.metrics import roc_auc_score

from src.validation.estimators import (
    aipw_ate,
    aipw_ate_cross_fitted,
    aipw_scores,
    aipw_scores_cross_fitted,
    apply_common_support_trim,
    attach_propensity,
    fit_propensity_score,
    fit_propensity_score_cross_fitted,
    get_matched_pairs,
    ipw_ate,
    naive_ols_ate,
    psm_ate,
    psm_pair_diffs,
    run_estimator_comparison,
    run_estimator_comparison_with_ci,
)


@pytest.fixture
def rct_data():
    rng = np.random.default_rng(0)
    n = 150_000
    treatment = rng.binomial(1, 0.85, n)
    f0 = rng.normal(0, 1, n)
    f1 = rng.normal(0, 1, n)

    true_ate = 0.03
    base_rate = 0.06
    p = np.clip(base_rate + true_ate * treatment + 0.03 * f0, 0.001, 0.999)
    visit = rng.binomial(1, p)

    df = pd.DataFrame({"f0": f0, "f1": f1, "treatment": treatment, "visit": visit})
    return df, true_ate


@pytest.fixture
def small_data():
    rng = np.random.default_rng(10)
    n = 30_000
    treatment = rng.binomial(1, 0.85, n)
    f0 = rng.normal(0, 1, n)
    f1 = rng.normal(0, 1, n)
    visit = rng.binomial(1, np.clip(0.06 + 0.03 * treatment + 0.03 * f0, 0.001, 0.999))
    return pd.DataFrame({"f0": f0, "f1": f1, "treatment": treatment, "visit": visit})


def _with_ps(df, cols=("f0", "f1")):
    df = df.copy()
    df["_propensity"] = fit_propensity_score(df, "treatment", list(cols))
    return df


def test_naive_ols_matches_diff_in_means_without_covariates(small_data):
    df = small_data
    ols_estimate = naive_ols_ate(df, "visit", "treatment")

    diff_in_means = df.loc[df.treatment == 1, "visit"].mean() - df.loc[df.treatment == 0, "visit"].mean()

    assert ols_estimate == pytest.approx(diff_in_means, abs=1e-9)


def test_fit_propensity_score_is_clipped_away_from_zero_and_one(small_data):
    propensity = fit_propensity_score(small_data, "treatment", ["f0", "f1"])

    assert propensity.min() >= 1e-3
    assert propensity.max() <= 1 - 1e-3


def test_attach_propensity_adds_column_without_mutating_input(small_data):
    out = attach_propensity(small_data, "treatment", ["f0", "f1"], cross_fit=True, random_state=0)

    assert "_propensity" in out.columns
    assert "_propensity" not in small_data.columns
    assert len(out) == len(small_data)


def test_common_support_trim_overlap_keeps_only_shared_region(small_data):
    df = _with_ps(small_data)

    trimmed = apply_common_support_trim(df, "_propensity", "treatment", method="overlap")

    treated_ps = df.loc[df.treatment == 1, "_propensity"]
    control_ps = df.loc[df.treatment == 0, "_propensity"]
    expected_lower = max(treated_ps.min(), control_ps.min())
    expected_upper = min(treated_ps.max(), control_ps.max())

    assert trimmed["_propensity"].min() >= expected_lower
    assert trimmed["_propensity"].max() <= expected_upper
    assert len(trimmed) <= len(df)


def test_common_support_trim_fixed_band_respects_bounds(small_data):
    df = _with_ps(small_data)

    band = (df["_propensity"].quantile(0.05), df["_propensity"].quantile(0.95))
    trimmed = apply_common_support_trim(df, "_propensity", "treatment", method="fixed", fixed_bounds=band)

    assert len(trimmed) > 0
    assert trimmed["_propensity"].min() >= band[0]
    assert trimmed["_propensity"].max() <= band[1]


def test_common_support_trim_rejects_unknown_method(small_data):
    df = _with_ps(small_data)
    with pytest.raises(ValueError):
        apply_common_support_trim(df, "_propensity", "treatment", method="bogus")


def test_get_matched_pairs_produces_equal_treated_and_control_counts(small_data):
    df = _with_ps(small_data)

    matched = get_matched_pairs(df, "treatment", "_propensity", caliper=0.2, random_state=0)

    n_treated = (matched["treatment"] == 1).sum()
    n_control = (matched["treatment"] == 0).sum()
    assert n_treated == n_control
    assert matched["_pair_id"].nunique() == n_treated


def test_get_matched_pairs_improves_covariate_balance():
    rng = np.random.default_rng(2)
    n = 20_000
    f0 = rng.normal(0, 1, n)
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


def test_get_matched_pairs_caliper_is_applied_on_the_logit_scale():
    rng = np.random.default_rng(6)
    n = 4_000
    treatment = rng.binomial(1, 0.5, n)
    propensity = np.clip(rng.beta(8, 1, n), 1e-3, 1 - 1e-3)
    df = pd.DataFrame({"treatment": treatment, "_propensity": propensity, "visit": 0})

    caliper = 0.2
    matched = get_matched_pairs(df, "treatment", "_propensity", caliper=caliper, random_state=0)

    logit = np.log(propensity / (1 - propensity))
    max_allowed = caliper * logit.std()
    m = matched.assign(_logit=np.log(matched["_propensity"] / (1 - matched["_propensity"])))
    treated = m[m.treatment == 1].set_index("_pair_id")["_logit"]
    control = m[m.treatment == 0].set_index("_pair_id")["_logit"]

    assert len(treated) > 0
    assert (treated - control.loc[treated.index]).abs().max() <= max_allowed + 1e-9


def test_get_matched_pairs_uses_each_control_at_most_once(small_data):
    df = _with_ps(small_data)
    df["_row_id"] = np.arange(len(df))

    matched = get_matched_pairs(df, "treatment", "_propensity", caliper=0.2, random_state=0)

    n_pairs = matched["_pair_id"].nunique()
    n_unique_controls = matched.loc[matched.treatment == 0, "_row_id"].nunique()

    assert n_unique_controls == n_pairs


def test_get_matched_pairs_caps_matches_at_size_of_minority_group(small_data):
    df = _with_ps(small_data)
    n_control = (df["treatment"] == 0).sum()

    matched = get_matched_pairs(df, "treatment", "_propensity", caliper=0.2, random_state=0)

    assert matched["_pair_id"].nunique() <= n_control


def test_psm_pair_diffs_mean_matches_psm_ate(small_data):
    df = _with_ps(small_data)

    matched = get_matched_pairs(df, "treatment", "_propensity", caliper=0.2, random_state=1)
    diffs = psm_pair_diffs(matched, "visit", "treatment")
    direct_estimate = psm_ate(df, "visit", "treatment", "_propensity", caliper=0.2, random_state=1)

    assert diffs.mean() == pytest.approx(direct_estimate, abs=1e-9)


def test_ipw_ate_is_exactly_zero_for_a_constant_outcome():
    rng = np.random.default_rng(3)
    n = 10_000
    treatment = rng.binomial(1, 0.85, n)
    propensity = np.clip(rng.beta(2, 8, n), 1e-3, 1 - 1e-3)
    df = pd.DataFrame({"treatment": treatment, "visit": np.ones(n), "_propensity": propensity})

    assert ipw_ate(df, "visit", "treatment", "_propensity") == pytest.approx(0.0, abs=1e-12)


def test_ipw_ate_recovers_a_known_effect_under_known_propensity():
    rng = np.random.default_rng(4)
    n = 400_000
    x = rng.normal(0, 1, n)
    propensity = 1 / (1 + np.exp(-(1.0 + 0.8 * x)))
    treatment = rng.binomial(1, propensity)
    base = 1 / (1 + np.exp(-(-2.0 + 0.8 * x)))
    visit = rng.binomial(1, np.clip(base + 0.05 * treatment, 0.001, 0.999))
    df = pd.DataFrame({"treatment": treatment, "visit": visit, "_propensity": propensity})

    naive = df.loc[df.treatment == 1, "visit"].mean() - df.loc[df.treatment == 0, "visit"].mean()
    adjusted = ipw_ate(df, "visit", "treatment", "_propensity")

    assert abs(naive - 0.05) > 0.01
    assert adjusted == pytest.approx(0.05, abs=0.006)


def test_aipw_runs_and_returns_a_finite_scalar(small_data):
    df = _with_ps(small_data)

    estimate = aipw_ate(df, "visit", "treatment", ["f0", "f1"], "_propensity")

    assert np.isfinite(estimate)


def test_aipw_scores_mean_matches_aipw_ate(small_data):
    df = _with_ps(small_data)

    scores = aipw_scores(df, "visit", "treatment", ["f0", "f1"], "_propensity")
    estimate = aipw_ate(df, "visit", "treatment", ["f0", "f1"], "_propensity")

    assert scores.mean() == pytest.approx(estimate, abs=1e-9)


def test_cross_fitted_propensity_does_not_overfit_noise_unlike_in_sample():
    rng = np.random.default_rng(7)
    n, k = 600, 100
    X = rng.normal(0, 1, (n, k))
    treatment = rng.binomial(1, 0.5, n)
    cols = [f"f{i}" for i in range(k)]
    df = pd.DataFrame(X, columns=cols)
    df["treatment"] = treatment

    in_sample = fit_propensity_score(df, "treatment", cols)
    cross_fitted = fit_propensity_score_cross_fitted(df, "treatment", cols, n_splits=5, random_state=0)

    in_sample_auc = roc_auc_score(treatment, in_sample)
    cross_fitted_auc = roc_auc_score(treatment, cross_fitted)

    assert in_sample_auc > 0.68
    assert cross_fitted_auc < 0.56
    assert in_sample_auc - cross_fitted_auc > 0.15


def test_cross_fitted_propensity_is_clipped_and_full_length(small_data):
    propensity = fit_propensity_score_cross_fitted(small_data, "treatment", ["f0", "f1"], n_splits=5, random_state=0)

    assert len(propensity) == len(small_data)
    assert propensity.min() >= 1e-3
    assert propensity.max() <= 1 - 1e-3


def test_aipw_cross_fitted_uses_the_supplied_propensity_column(small_data):
    base = attach_propensity(small_data, "treatment", ["f0", "f1"], cross_fit=True, random_state=0)
    shifted = base.copy()
    shifted["_propensity"] = np.clip(shifted["_propensity"] * 0.9, 1e-3, 1 - 1e-3)

    a = aipw_ate_cross_fitted(base, "visit", "treatment", ["f0", "f1"], "_propensity", random_state=0)
    b = aipw_ate_cross_fitted(shifted, "visit", "treatment", ["f0", "f1"], "_propensity", random_state=0)

    assert a != pytest.approx(b, abs=1e-9)


def test_aipw_cross_fitted_scores_mean_matches_aipw_ate_cross_fitted(small_data):
    df = attach_propensity(small_data, "treatment", ["f0", "f1"], cross_fit=True, random_state=0)

    scores = aipw_scores_cross_fitted(df, "visit", "treatment", ["f0", "f1"], "_propensity", random_state=0)
    estimate = aipw_ate_cross_fitted(df, "visit", "treatment", ["f0", "f1"], "_propensity", random_state=0)

    assert scores.mean() == pytest.approx(estimate, abs=1e-9)


def test_run_estimator_comparison_returns_all_four_methods(small_data):
    results = run_estimator_comparison(small_data, "visit", "treatment", ["f0", "f1"], random_state=0)

    assert set(results.keys()) == {"naive_ols", "psm", "ipw", "aipw"}
    for estimate in results.values():
        assert np.isfinite(estimate)


def test_all_estimators_recover_true_ate_on_rct_data(rct_data):
    df, true_ate = rct_data
    results = run_estimator_comparison(df, "visit", "treatment", ["f0", "f1"], random_state=0)

    for method, estimate in results.items():
        assert estimate == pytest.approx(true_ate, abs=0.006), f"{method} estimate {estimate} too far from {true_ate}"


def test_run_estimator_comparison_cross_fit_toggle_produces_different_aipw_estimate(small_data):
    with_cross_fit = run_estimator_comparison(small_data, "visit", "treatment", ["f0", "f1"], cross_fit=True, random_state=0)
    without_cross_fit = run_estimator_comparison(small_data, "visit", "treatment", ["f0", "f1"], cross_fit=False, random_state=0)

    assert with_cross_fit["aipw"] != pytest.approx(without_cross_fit["aipw"], abs=1e-9)


def test_run_estimator_comparison_with_ci_produces_valid_intervals(small_data):
    results = run_estimator_comparison_with_ci(
        small_data, "visit", "treatment", ["f0", "f1"], n_bootstrap=50, random_state=0
    )

    assert set(results.keys()) == {"naive_ols", "psm", "ipw", "aipw"}
    for method, r in results.items():
        assert r["ci_lower"] <= r["point_estimate"] <= r["ci_upper"], method
        assert r["ci_upper"] > r["ci_lower"], method
