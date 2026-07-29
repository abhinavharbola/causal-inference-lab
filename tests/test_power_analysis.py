import numpy as np
import pytest

from src.utils.power_analysis import (
    calculate_mde,
    mde_comparison_table,
    required_sample_size,
    power_curve,
    segment_power_analysis,
)


def test_calculate_mde_returns_smaller_effect_for_larger_n():
    small_n = calculate_mde("visit", baseline_rate=0.045, n_per_group=10_000)
    large_n = calculate_mde("visit", baseline_rate=0.045, n_per_group=200_000)

    assert large_n.mde_absolute < small_n.mde_absolute


def test_conversion_requires_larger_relative_effect_than_visit_at_same_n():
    visit_result = calculate_mde("visit", baseline_rate=0.045, n_per_group=100_000)
    conversion_result = calculate_mde("conversion", baseline_rate=0.003, n_per_group=100_000)

    assert conversion_result.mde_relative > visit_result.mde_relative


def test_mde_comparison_table_has_expected_columns_and_row_count():
    table = mde_comparison_table({"visit": 0.045, "conversion": 0.003}, n_per_group=50_000)

    assert len(table) == 2
    assert set(table["outcome"]) == {"visit", "conversion"}
    for col in ["baseline_rate", "n_per_group", "mde_absolute", "mde_relative_pct"]:
        assert col in table.columns


def test_required_sample_size_is_consistent_with_calculate_mde():
    baseline_rate = 0.045
    true_effect = 0.005

    required_n = required_sample_size(baseline_rate, true_effect)
    mde_at_required_n = calculate_mde("visit", baseline_rate, n_per_group=required_n)

    assert mde_at_required_n.mde_absolute == pytest.approx(true_effect, rel=0.05)


def test_power_curve_is_monotonically_increasing_in_n():
    n_range = np.array([1_000, 5_000, 20_000, 100_000])
    curve = power_curve(baseline_rate=0.045, true_effect_absolute=0.005, n_range=n_range)

    powers = curve["power"].to_numpy()
    assert np.all(np.diff(powers) >= 0)
    assert powers[-1] > powers[0]


def test_segment_power_analysis_flags_small_segments_as_underpowered():
    segments = {"large": 200_000, "tiny": 500}
    result = segment_power_analysis(
        segments, baseline_rate=0.045, true_effect_absolute=0.003, treatment_share=0.85
    )

    tiny_row = result[result["segment"] == "tiny"].iloc[0]
    large_row = result[result["segment"] == "large"].iloc[0]

    assert tiny_row["underpowered"]
    assert tiny_row["achieved_power"] < large_row["achieved_power"]


def test_segment_power_analysis_respects_treatment_share():
    segments = {"seg": 100_000}

    balanced = segment_power_analysis(segments, baseline_rate=0.045, true_effect_absolute=0.005, treatment_share=0.5)
    imbalanced = segment_power_analysis(segments, baseline_rate=0.045, true_effect_absolute=0.005, treatment_share=0.85)

    assert imbalanced.iloc[0]["achieved_power"] < balanced.iloc[0]["achieved_power"]


def test_segment_power_analysis_handles_zero_arm_segment_gracefully():
    segments = {"empty_control_risk": 1}
    result = segment_power_analysis(segments, baseline_rate=0.045, true_effect_absolute=0.005, treatment_share=0.99)

    assert result.iloc[0]["underpowered"]
    assert result.iloc[0]["achieved_power"] == 0.0