"""
Shared power-analysis machinery, used in three places across the project:

1. Section 1.5: MDE comparison between `visit` and `conversion` at the
   planned Section 2 subsample size, to justify the outcome variable choice.
2. Section 1: power curves for the validation pipeline, characterizing
   how much data the estimator comparison needs to be trustworthy.
3. Section 3: per-segment power analysis, to show whether naive per-segment
   tests are underpowered once the sample is split into clusters.

All three reuse the same NormalIndPower / proportion_effectsize machinery,
no duplicated stats code between sections.
"""

import logging
import math
from dataclasses import dataclass

import numpy as np
import pandas as pd
from statsmodels.stats.power import NormalIndPower
from statsmodels.stats.proportion import proportion_effectsize

logger = logging.getLogger(__name__)

_power_calc = NormalIndPower()


@dataclass
class MDEResult:
    outcome_name: str
    baseline_rate: float
    n_per_group: int
    alpha: float
    power: float
    effect_size_h: float          # Cohen's h, standardized effect size
    mde_absolute: float           # minimum detectable absolute rate difference
    mde_relative: float           # mde_absolute / baseline_rate


def _effect_size_to_absolute_mde(baseline_rate: float, effect_size_h: float) -> float:
    """
    Inverts Cohen's h back to an absolute proportion difference given a
    fixed baseline rate. h = 2*arcsin(sqrt(p1)) - 2*arcsin(sqrt(p2)).

    solve_power's reverse solve (given n, alpha, power) always returns a
    positive effect-size magnitude, regardless of direction. Since this
    project cares about the uplift direction (treatment increases
    visit/conversion probability), the positive h must be ADDED to phi1
    to recover the increase direction, not subtracted: subtracting would
    silently compute the decrease direction instead, which — because the
    arcsine transform is nonlinear — is not the same absolute magnitude
    as the increase direction, especially at low base rates. Confirmed
    during testing: subtracting produced an MDE of 0.00475 for a true
    0.005 effect at a 4.5% baseline, a systematic ~5% understatement.
    """
    phi1 = 2 * math.asin(math.sqrt(baseline_rate))
    phi2 = phi1 + effect_size_h
    p2 = math.sin(phi2 / 2) ** 2
    return abs(p2 - baseline_rate)


def calculate_mde(
    outcome_name: str,
    baseline_rate: float,
    n_per_group: int,
    alpha: float = 0.05,
    power: float = 0.8,
) -> MDEResult:
    """
    Given a baseline rate and a fixed sample size per arm, back-calculates
    the minimum detectable effect (MDE) at the given alpha/power.
    """
    effect_size_h = _power_calc.solve_power(
        effect_size=None,
        nobs1=n_per_group,
        alpha=alpha,
        power=power,
        ratio=1.0,
        alternative="two-sided",
    )

    mde_absolute = _effect_size_to_absolute_mde(baseline_rate, effect_size_h)
    mde_relative = mde_absolute / baseline_rate if baseline_rate > 0 else float("inf")

    return MDEResult(
        outcome_name=outcome_name,
        baseline_rate=baseline_rate,
        n_per_group=n_per_group,
        alpha=alpha,
        power=power,
        effect_size_h=effect_size_h,
        mde_absolute=mde_absolute,
        mde_relative=mde_relative,
    )


def mde_comparison_table(
    baseline_rates: dict,
    n_per_group: int,
    alpha: float = 0.05,
    power: float = 0.8,
) -> pd.DataFrame:
    """
    Builds the Section 1.5 comparison table: for each outcome (e.g.
    'visit', 'conversion'), what effect size can be detected at the
    planned subsample size. This is the artifact that justifies which
    outcome Sections 2 and 3 use.
    """
    rows = []
    for outcome_name, baseline_rate in baseline_rates.items():
        result = calculate_mde(outcome_name, baseline_rate, n_per_group, alpha, power)
        rows.append(
            {
                "outcome": result.outcome_name,
                "baseline_rate": result.baseline_rate,
                "n_per_group": result.n_per_group,
                "alpha": result.alpha,
                "power": result.power,
                "mde_absolute": result.mde_absolute,
                "mde_relative_pct": result.mde_relative * 100,
            }
        )
        logger.info(
            "MDE for %s: absolute=%.5f, relative=%.2f%% at n=%d",
            outcome_name,
            result.mde_absolute,
            result.mde_relative * 100,
            n_per_group,
        )
    return pd.DataFrame(rows)


def required_sample_size(
    baseline_rate: float,
    true_effect_absolute: float,
    alpha: float = 0.05,
    power: float = 0.8,
) -> int:
    """
    Back-calculates the required per-arm sample size to detect a known
    true effect size at the given alpha/power. Used in Section 1 to show
    how much data the validation pipeline needs to reliably distinguish
    naive bias from the ground-truth effect.
    """
    p1 = baseline_rate
    p2 = baseline_rate + true_effect_absolute
    effect_size_h = proportion_effectsize(p1, p2)

    n = _power_calc.solve_power(
        effect_size=effect_size_h,
        nobs1=None,
        alpha=alpha,
        power=power,
        ratio=1.0,
        alternative="two-sided",
    )
    return int(math.ceil(n))


def power_curve(
    baseline_rate: float,
    true_effect_absolute: float,
    n_range: np.ndarray,
    alpha: float = 0.05,
) -> pd.DataFrame:
    """
    Computes achieved statistical power across a range of sample sizes,
    for a fixed true effect. Used in Section 1 to visualize where the
    estimator comparison starts to have adequate power, and referenced
    by the Section 1.5 MDE table as the same underlying tool used for
    a different purpose (n->power here, n->effect there).
    """
    p1 = baseline_rate
    p2 = baseline_rate + true_effect_absolute
    effect_size_h = proportion_effectsize(p1, p2)

    powers = []
    for n in n_range:
        achieved_power = _power_calc.solve_power(
            effect_size=effect_size_h,
            nobs1=int(n),
            alpha=alpha,
            power=None,
            ratio=1.0,
            alternative="two-sided",
        )
        powers.append(achieved_power)

    return pd.DataFrame({"n_per_group": n_range, "power": powers})


def segment_power_analysis(
    segment_sizes: dict,
    baseline_rate: float,
    true_effect_absolute: float,
    treatment_share: float = 0.85,
    alpha: float = 0.05,
    power_threshold: float = 0.8,
) -> pd.DataFrame:
    """
    Section 3: given per-segment sample sizes (after splitting the full
    sample into clusters/business segments), computes achieved power for
    each segment at the known or assumed true effect size, and flags
    segments that are underpowered. This is meant to surface the common
    real-world mistake of running per-segment tests without checking
    whether the segment is even large enough to detect the effect.

    treatment_share: fraction of each segment in the treatment arm.
    Criteo's actual treatment/control split is closer to 85/15 than 50/50
    (see data_loader.EXPECTED_TREATMENT_SHARE_*), so this is parameterized
    rather than assumed balanced. Unequal arm sizes reduce achieved power
    relative to a balanced split at the same total n, which is exactly
    the kind of thing this check exists to catch.
    """
    p1 = baseline_rate
    p2 = baseline_rate + true_effect_absolute
    effect_size_h = proportion_effectsize(p1, p2)

    rows = []
    for segment_name, n_segment in segment_sizes.items():
        n_treatment = int(round(n_segment * treatment_share))
        n_control = n_segment - n_treatment

        if n_control <= 0 or n_treatment <= 0:
            logger.warning(
                "Segment %s has zero units in one arm (n_treatment=%d, n_control=%d), "
                "skipping power calculation",
                segment_name,
                n_treatment,
                n_control,
            )
            rows.append(
                {
                    "segment": segment_name,
                    "n_segment": n_segment,
                    "n_treatment": n_treatment,
                    "n_control": n_control,
                    "achieved_power": 0.0,
                    "underpowered": True,
                }
            )
            continue

        # statsmodels convention: nobs1 is the reference group, ratio = nobs2 / nobs1.
        # Control is used as the reference group here.
        ratio = n_treatment / n_control

        achieved_power = _power_calc.solve_power(
            effect_size=effect_size_h,
            nobs1=n_control,
            alpha=alpha,
            power=None,
            ratio=ratio,
            alternative="two-sided",
        )
        rows.append(
            {
                "segment": segment_name,
                "n_segment": n_segment,
                "n_treatment": n_treatment,
                "n_control": n_control,
                "achieved_power": achieved_power,
                "underpowered": achieved_power < power_threshold,
            }
        )

    df = pd.DataFrame(rows)
    n_underpowered = df["underpowered"].sum()
    if n_underpowered > 0:
        logger.warning(
            "%d of %d segments are underpowered (power < %.2f) at effect size %.5f",
            n_underpowered,
            len(df),
            power_threshold,
            true_effect_absolute,
        )
    return df


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)

    baseline_rates = {"visit": 0.045, "conversion": 0.003}
    planned_n_per_group = 100_000

    table = mde_comparison_table(baseline_rates, planned_n_per_group)
    print(table)