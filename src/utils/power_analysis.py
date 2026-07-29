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