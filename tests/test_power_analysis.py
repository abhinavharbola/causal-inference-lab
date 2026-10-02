import numpy as np
import pytest

from src.utils.power_analysis import (
    calculate_mde,
    mde_comparison_table,
    power_curve,
    required_sample_size,
    segment_power_analysis,
)


def test_calculate_mde_shrinks_as_sample_size_grows():
    small = calculate_mde("visit", 0.038, 10_000)
    large = calculate_mde("visit", 0.038, 100_000)

    assert large.mde_absolute < small.mde_absolute


def test_calculate_mde_relative_is_larger_for_rarer_outcomes():
    visit = calculate_mde("visit", 0.038, 50_000)
    conversion = calculate_mde("conversion", 0.002, 50_000)

    assert conversion.mde_relative > visit.mde_relative


def test_calculate_mde_matches_known_reference_value():
    result = calculate_mde("visit", 0.046992, 150_000)

    assert result.mde_absolute == pytest.approx(0.002189, abs=2e-5)
    assert result.mde_relative * 100 == pytest.approx(4.66, abs=0.05)


def test_calculate_mde_imbalanced_arms_are_more_sensitive_than_equal_control_only():
    balanced = calculate_mde("visit", 0.038, 20_000, ratio=1.0)
    imbalanced = calculate_mde("visit", 0.038, 20_000, ratio=5.0)

    assert imbalanced.mde_absolute < balanced.mde_absolute


def test_calculate_mde_imbalanced_arms_are_less_sensitive_than_equal_total_size():
    n_total = 150_000
    equal = calculate_mde("visit", 0.038, n_total // 2, ratio=1.0)
    n_control = int(n_total * 0.15)
    skewed = calculate_mde("visit", 0.038, n_control, ratio=(n_total - n_control) / n_control)

    assert skewed.mde_absolute > equal.mde_absolute


def test_mde_comparison_table_has_one_row_per_outcome_and_records_ratio():
    table = mde_comparison_table({"visit": 0.038, "conversion": 0.002}, 22_500, ratio=5.67)

    assert list(table["outcome"]) == ["visit", "conversion"]
    assert (table["ratio"] == 5.67).all()
    assert (table["mde_absolute"] > 0).all()


def test_required_sample_size_decreases_for_larger_effects():
    small_effect = required_sample_size(0.038, 0.005)
    large_effect = required_sample_size(0.038, 0.02)

    assert large_effect < small_effect


def test_required_sample_size_is_smaller_per_control_unit_with_more_treated_units():
    balanced = required_sample_size(0.038, 0.01, ratio=1.0)
    skewed = required_sample_size(0.038, 0.01, ratio=5.0)

    assert skewed < balanced


def test_power_curve_is_monotonically_increasing_in_n():
    curve = power_curve(0.038, 0.01, np.array([1_000, 5_000, 20_000, 100_000]))

    assert curve["power"].is_monotonic_increasing


def test_segment_power_analysis_flags_small_segments_as_underpowered():
    arms = {"big": (85_000, 15_000), "tiny": (85, 15)}
    table = segment_power_analysis(arms, baseline_rate=0.038, true_effect_absolute=0.01).set_index("segment")

    assert not table.loc["big", "underpowered"]
    assert table.loc["tiny", "underpowered"]
    assert table.loc["big", "achieved_power"] > table.loc["tiny", "achieved_power"]


def test_segment_power_analysis_uses_the_observed_arm_sizes():
    arms = {"even": (5_000, 5_000), "skewed": (9_000, 1_000)}
    table = segment_power_analysis(arms, baseline_rate=0.038, true_effect_absolute=0.01).set_index("segment")

    assert table.loc["even", "n_control"] == 5_000
    assert table.loc["skewed", "n_treatment"] == 9_000
    assert table.loc["even", "achieved_power"] > table.loc["skewed", "achieved_power"]


def test_segment_power_analysis_handles_an_empty_arm_without_raising():
    table = segment_power_analysis({"empty": (100, 0)}, baseline_rate=0.038, true_effect_absolute=0.01)

    assert table.loc[0, "underpowered"]
    assert table.loc[0, "achieved_power"] == 0.0
