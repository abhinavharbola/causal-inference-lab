import numpy as np
import pandas as pd
import pytest

from src.validation.estimators import fit_propensity_score
from src.validation.diagnostics import run_full_diagnostics


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


def test_run_full_diagnostics_reports_consistent_match_counts(confounded_data):
    result = run_full_diagnostics(
        confounded_data, ["f0", "f1"], "treatment", "_propensity", caliper=0.2, random_state=0
    )

    assert result["n_pairs"] <= result["n_treated_total"]
    assert 0.0 <= result["match_rate"] <= 1.0
    # Each matched pair should draw on a distinct control row (see get_matched_pairs).
    assert result["n_control_unique"] == result["n_pairs"]


def test_run_full_diagnostics_match_rate_reflects_minority_group_size(confounded_data):
    n_treated_total = (confounded_data["treatment"] == 1).sum()
    n_control_total = (confounded_data["treatment"] == 0).sum()

    result = run_full_diagnostics(
        confounded_data, ["f0", "f1"], "treatment", "_propensity", caliper=0.2, random_state=0
    )

    # Can't match more treated units than there are controls to draw from.
    assert result["n_pairs"] <= n_control_total
    assert result["match_rate"] == pytest.approx(result["n_pairs"] / n_treated_total)


def test_run_full_diagnostics_measures_overlap_on_prematch_population(confounded_data):
    # Regression test for a bug where overlap_diagnostics was run on the
    # already-matched subset instead of the pre-matching candidate pool. Since
    # matched pairs are selected specifically for being within a caliper of
    # each other, computing "pct_within_overlap" on them is close to
    # tautological (it reads ~100% almost regardless of the real population's
    # overlap) and feeds a misleading "overlap is strong" signal into the LLM
    # critique. It must be computed on the full pre-matching population.
    result = run_full_diagnostics(
        confounded_data, ["f0", "f1"], "treatment", "_propensity", caliper=0.2, random_state=0
    )

    assert result["overlap"]["n_total"] == len(confounded_data)



    # Regression fixture for a bug where n_control_unique was computed by
    # deduplicating on covariate_cols instead of row identity. Criteo's
    # anonymized f-columns are bucketed/discretized in production, so distinct
    # control rows commonly share identical covariate values; this fixture
    # reproduces that on purpose (only 6 distinct f0 values across 20k rows)
    # so a covariate-based dedup would undercount n_control_unique even though
    # get_matched_pairs never reuses a control row.
    rng = np.random.default_rng(5)
    n = 20_000
    f0 = rng.choice(np.arange(6, dtype=float), size=n)
    treat_prob = 1 / (1 + np.exp(-0.4 * f0))
    treatment = rng.binomial(1, treat_prob)
    visit = rng.binomial(1, np.clip(0.045 + 0.01 * treatment, 0.001, 0.999))
    df = pd.DataFrame({"f0": f0, "treatment": treatment, "visit": visit})
    df["_propensity"] = fit_propensity_score(df, "treatment", ["f0"])
    return df


def test_run_full_diagnostics_counts_control_rows_not_covariate_values(bucketed_confounded_data):
    result = run_full_diagnostics(
        bucketed_confounded_data, ["f0"], "treatment", "_propensity", caliper=0.2, random_state=0
    )
    matched_controls = result["matched_df"].loc[result["matched_df"]["treatment"] == 0]

    assert result["n_pairs"] > 0
    # Confirm the fixture actually exercises the bug scenario: matched control
    # rows repeat covariate values, so a covariate-based dedup undercounts them.
    assert matched_controls["f0"].duplicated().any()
    assert matched_controls["f0"].nunique() < len(matched_controls)

    # n_control_unique must still equal n_pairs: get_matched_pairs matches 1:1
    # without replacement regardless of whether covariate values repeat.
    assert result["n_control_unique"] == result["n_pairs"]


