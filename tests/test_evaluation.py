import numpy as np
import pandas as pd

from src.heterogeneity.evaluation import apply_benjamini_hochberg, evaluate_cate_qini


def test_benjamini_hochberg_adjusts_monotonically_and_preserves_attrs():
    df = pd.DataFrame({"segment": list("abcd"), "p_value": [0.001, 0.02, 0.03, 0.6]})
    df.attrs["marker"] = {"kept": True}

    result = apply_benjamini_hochberg(df)

    assert result.attrs["marker"] == {"kept": True}
    assert (result["p_value_adjusted"] >= result["p_value"]).all()
    assert list(result["significant_after_correction"]) == [True, True, True, False]


def test_benjamini_hochberg_can_remove_a_marginal_raw_significant_segment():
    df = pd.DataFrame({"segment": list("abcd"), "p_value": [0.04, 0.045, 0.3, 0.8]})

    result = apply_benjamini_hochberg(df)

    assert result["significant_after_correction"].sum() < (df["p_value"] < 0.05).sum()


def test_evaluate_cate_qini_returns_ordered_interval_around_the_estimate():
    rng = np.random.default_rng(0)
    n = 6_000
    treatment = rng.binomial(1, 0.5, n)
    score = rng.normal(0, 1, n)
    outcome = rng.binomial(1, np.clip(0.1 + 0.15 * treatment * (score > 0), 0.001, 0.999))
    df = pd.DataFrame({"cate": score, "visit": outcome, "treatment": treatment})

    result = evaluate_cate_qini(df, "cate", "visit", "treatment", n_bootstrap=20, random_state=0)

    assert result["ci_lower"] <= result["ci_upper"]
    assert result["n_bootstrap"] == 20
    assert len(result["curve_x"]) == len(result["curve_y"])


def test_evaluate_cate_qini_can_skip_the_bootstrap():
    rng = np.random.default_rng(1)
    n = 2_000
    df = pd.DataFrame(
        {"cate": rng.normal(size=n), "visit": rng.binomial(1, 0.1, n), "treatment": rng.binomial(1, 0.5, n)}
    )

    result = evaluate_cate_qini(df, "cate", "visit", "treatment", n_bootstrap=0)

    assert np.isnan(result["ci_lower"]) and np.isnan(result["ci_upper"])
