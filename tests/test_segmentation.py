import numpy as np
import pandas as pd
import pytest

from src.heterogeneity.segmentation import (
    cluster_segments,
    compute_segment_effects,
    quantile_segments,
    segment_cate_separation,
    segment_effect_heterogeneity,
)


def test_segment_cate_separation_detects_real_cate_differences_across_segments():
    rng = np.random.default_rng(0)
    n_per_segment = 2_000

    segment_a = pd.DataFrame({"segment": "a", "cate": rng.normal(0.08, 0.01, n_per_segment)})
    segment_b = pd.DataFrame({"segment": "b", "cate": rng.normal(0.00, 0.01, n_per_segment)})
    df = pd.concat([segment_a, segment_b], ignore_index=True)

    result = segment_cate_separation(df, "segment", cate_col="cate")

    assert result["meaningfully_separated"]
    assert result["eta_squared"] > 0.5
    assert result["p_value"] < 0.001


def test_segment_cate_separation_flags_segments_with_no_real_cate_difference():
    rng = np.random.default_rng(1)
    n_per_segment = 2_000

    segment_a = pd.DataFrame({"segment": "a", "cate": rng.normal(0.02, 0.02, n_per_segment)})
    segment_b = pd.DataFrame({"segment": "b", "cate": rng.normal(0.02, 0.02, n_per_segment)})
    df = pd.concat([segment_a, segment_b], ignore_index=True)

    result = segment_cate_separation(df, "segment", cate_col="cate")

    assert not result["meaningfully_separated"]
    assert result["eta_squared"] < 0.05


def test_segment_cate_separation_handles_a_single_segment():
    df = pd.DataFrame({"segment": ["a"] * 100, "cate": np.random.default_rng(2).normal(0, 0.01, 100)})

    result = segment_cate_separation(df, "segment", cate_col="cate")

    assert not result["meaningfully_separated"]
    assert result["eta_squared"] == 0.0


def test_compute_segment_effects_attaches_cate_separation_when_cate_present():
    rng = np.random.default_rng(3)
    n = 4_000
    f0 = rng.normal(0, 1, n)
    treatment = rng.binomial(1, 0.85, n)
    visit = rng.binomial(1, np.clip(0.045 + 0.01 * treatment, 0.001, 0.999))
    cate = np.where(f0 > 0, 0.05, 0.00) + rng.normal(0, 0.01, n)

    df = pd.DataFrame({"f0": f0, "treatment": treatment, "visit": visit, "cate": cate})
    df["segment"] = cluster_segments(df, ["f0"], n_clusters=2, random_state=0)

    result_df = compute_segment_effects(df, "segment", "treatment", "visit", n_bootstrap=50, random_state=0)

    assert set(result_df.attrs["cate_separation"].keys()) == {
        "f_statistic", "p_value", "eta_squared", "meaningfully_separated",
    }
    assert set(result_df.attrs["effect_heterogeneity"].keys()) == {"q_statistic", "df", "p_value", "i_squared"}
    assert (result_df["se"] > 0).all()


def test_compute_segment_effects_skips_cate_separation_without_cate_column():
    rng = np.random.default_rng(4)
    n = 2_000
    treatment = rng.binomial(1, 0.85, n)
    visit = rng.binomial(1, np.clip(0.045 + 0.01 * treatment, 0.001, 0.999))
    segment = pd.Series(rng.choice(["a", "b"], size=n), name="segment")

    df = pd.DataFrame({"treatment": treatment, "visit": visit, "segment": segment})

    result_df = compute_segment_effects(df, "segment", "treatment", "visit", n_bootstrap=50, random_state=0)

    assert "cate_separation" not in result_df.attrs


def test_segment_effect_heterogeneity_detects_genuinely_different_effects():
    effects = pd.DataFrame({"point_estimate": [0.001, 0.05, 0.10], "se": [0.005, 0.005, 0.005]})

    result = segment_effect_heterogeneity(effects)

    assert result["p_value"] < 1e-6
    assert result["df"] == 2
    assert result["i_squared"] > 0.9


def test_segment_effect_heterogeneity_does_not_flag_identical_effects():
    effects = pd.DataFrame({"point_estimate": [0.010, 0.011, 0.009, 0.010], "se": [0.004] * 4})

    result = segment_effect_heterogeneity(effects)

    assert result["p_value"] > 0.5
    assert result["i_squared"] == 0.0


def test_segment_effect_heterogeneity_is_not_driven_by_nonzero_effects_alone():
    effects = pd.DataFrame({"point_estimate": [0.05, 0.05, 0.05], "se": [0.001, 0.001, 0.001]})

    assert segment_effect_heterogeneity(effects)["p_value"] == pytest.approx(1.0, abs=1e-6)


def test_segment_effect_heterogeneity_handles_a_single_segment():
    effects = pd.DataFrame({"point_estimate": [0.01], "se": [0.004]})

    assert np.isnan(segment_effect_heterogeneity(effects)["p_value"])


def test_quantile_segments_returns_labelled_quartiles_for_continuous_values():
    df = pd.DataFrame({"v": np.random.default_rng(0).normal(size=1_000)})

    segments = quantile_segments(df, "v", n_bins=4)

    assert set(segments.unique()) == {"q1", "q2", "q3", "q4"}
    assert segments.value_counts().min() >= 240


def test_quantile_segments_survives_duplicate_edges_with_default_labels():
    df = pd.DataFrame({"v": [0] * 70 + list(range(30))})

    segments = quantile_segments(df, "v", n_bins=4)

    assert segments.notna().all()
    assert 2 <= segments.nunique() < 4
    assert all(label.startswith("q") for label in segments.unique())


def test_quantile_segments_rejects_too_few_custom_labels():
    df = pd.DataFrame({"v": np.arange(100)})

    with pytest.raises(ValueError):
        quantile_segments(df, "v", n_bins=4, labels=["a", "b"])
