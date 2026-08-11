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
