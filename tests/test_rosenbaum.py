import pandas as pd
import pytest

from src.sensitivity.rosenbaum import (
    classify_gamma,
    count_discordant_pairs,
    find_critical_gamma,
    rosenbaum_bound_at_gamma,
)


def _pairs(n_plus: int, n_minus: int, n_concordant: int = 0) -> pd.DataFrame:
    rows = []
    pair_id = 0
    for _ in range(n_plus):
        rows += [
            {"_pair_id": pair_id, "treatment": 1, "visit": 1},
            {"_pair_id": pair_id, "treatment": 0, "visit": 0},
        ]
        pair_id += 1
    for _ in range(n_minus):
        rows += [
            {"_pair_id": pair_id, "treatment": 1, "visit": 0},
            {"_pair_id": pair_id, "treatment": 0, "visit": 1},
        ]
        pair_id += 1
    for _ in range(n_concordant):
        rows += [
            {"_pair_id": pair_id, "treatment": 1, "visit": 0},
            {"_pair_id": pair_id, "treatment": 0, "visit": 0},
        ]
        pair_id += 1
    return pd.DataFrame(rows)


def test_count_discordant_pairs_splits_plus_minus_and_concordant():
    counts = count_discordant_pairs(_pairs(7, 3, 5), "visit", "treatment")

    assert counts["n_plus"] == 7
    assert counts["n_minus"] == 3
    assert counts["n_concordant"] == 5
    assert counts["n_discordant"] == 10
    assert counts["n_pairs"] == 15


def test_rosenbaum_bound_is_monotone_in_gamma():
    p_values = [rosenbaum_bound_at_gamma(70, 100, g) for g in (1.0, 1.5, 2.0, 3.0)]

    assert p_values == sorted(p_values)


def test_rosenbaum_bound_returns_one_without_discordant_pairs():
    assert rosenbaum_bound_at_gamma(0, 0, 1.0) == 1.0


def test_find_critical_gamma_solves_the_bound_exactly():
    matched = _pairs(700, 500)
    result = find_critical_gamma(matched, "visit", "treatment", alpha=0.05)

    assert result["significant_at_gamma_1"]
    assert result["critical_gamma"] > 1.0
    at_critical = rosenbaum_bound_at_gamma(result["n_plus"], result["n_discordant"], result["critical_gamma"])
    assert at_critical == pytest.approx(0.05, abs=1e-6)


def test_find_critical_gamma_flags_a_result_that_is_not_significant_at_gamma_one():
    matched = _pairs(52, 48)
    result = find_critical_gamma(matched, "visit", "treatment", alpha=0.05)

    assert not result["significant_at_gamma_1"]
    assert result["critical_gamma"] is None
    assert classify_gamma(result) == "not_significant"


def test_find_critical_gamma_reports_none_when_robust_beyond_gamma_max():
    matched = _pairs(2_000, 5)
    result = find_critical_gamma(matched, "visit", "treatment", alpha=0.05, gamma_max=3.0)

    assert result["significant_at_gamma_1"]
    assert result["critical_gamma"] is None
    assert classify_gamma(result) == "robust"


def test_find_critical_gamma_handles_no_discordant_pairs():
    result = find_critical_gamma(_pairs(0, 0, 50), "visit", "treatment")

    assert not result["significant_at_gamma_1"]
    assert result["critical_gamma"] is None


def test_classify_gamma_thresholds():
    base = {"significant_at_gamma_1": True}

    assert classify_gamma({**base, "critical_gamma": 1.2}) == "fragile"
    assert classify_gamma({**base, "critical_gamma": 2.0}) == "moderate"
    assert classify_gamma({**base, "critical_gamma": 3.5}) == "robust"
    assert classify_gamma({**base, "critical_gamma": None}) == "robust"
