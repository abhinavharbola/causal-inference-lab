import logging

import pandas as pd

logger = logging.getLogger(__name__)

EXPECTED_FEATURE_COLUMNS = [f"f{i}" for i in range(12)]
EXPECTED_COLUMNS = EXPECTED_FEATURE_COLUMNS + ["treatment", "exposure", "conversion", "visit"]

EXPECTED_ROW_COUNT = 13_979_592

EXPECTED_TREATMENT_SHARE = 0.85
TREATMENT_SHARE_TOLERANCE = 0.005

EXPECTED_VISIT_RATE = 0.0470
VISIT_RATE_TOLERANCE = 0.001

EXPECTED_CONVERSION_RATE = 0.0029
CONVERSION_RATE_TOLERANCE = 0.0002


class DataIntegrityError(Exception):
    pass


def _check_columns(df: pd.DataFrame) -> None:
    missing = set(EXPECTED_COLUMNS) - set(df.columns)
    extra = set(df.columns) - set(EXPECTED_COLUMNS)
    if missing:
        raise DataIntegrityError(f"Missing expected columns: {sorted(missing)}")
    if extra:
        raise DataIntegrityError(f"Unexpected extra columns: {sorted(extra)}")


def _check_row_count(df: pd.DataFrame) -> None:
    n = len(df)
    if n != EXPECTED_ROW_COUNT:
        raise DataIntegrityError(f"Row count {n} does not match documented {EXPECTED_ROW_COUNT}")
    logger.info("Row count check passed: %d rows", n)


def _check_within(name: str, value: float, expected: float, tolerance: float) -> None:
    if abs(value - expected) > tolerance:
        raise DataIntegrityError(
            f"{name} {value:.5f} deviates from documented {expected:.5f} by more than {tolerance:.5f}"
        )


def _check_treatment_split(df: pd.DataFrame) -> None:
    share = df["treatment"].mean()
    _check_within("Treatment share", share, EXPECTED_TREATMENT_SHARE, TREATMENT_SHARE_TOLERANCE)
    logger.info("Treatment split check passed: %.4f treated", share)


def _check_outcome_rates(df: pd.DataFrame) -> None:
    visit_rate = df["visit"].mean()
    conversion_rate = df["conversion"].mean()
    _check_within("Visit rate", visit_rate, EXPECTED_VISIT_RATE, VISIT_RATE_TOLERANCE)
    _check_within("Conversion rate", conversion_rate, EXPECTED_CONVERSION_RATE, CONVERSION_RATE_TOLERANCE)
    logger.info("Outcome rate check passed: visit=%.4f, conversion=%.4f", visit_rate, conversion_rate)


def run_integrity_check(df: pd.DataFrame) -> None:
    _check_columns(df)
    _check_row_count(df)
    _check_treatment_split(df)
    _check_outcome_rates(df)


def _load_from_huggingface() -> pd.DataFrame:
    from datasets import load_dataset

    ds = load_dataset("criteo/criteo-uplift", split="train")
    return ds.to_pandas()


def _load_from_sklift() -> pd.DataFrame:
    from sklift.datasets import fetch_criteo

    bunch = fetch_criteo(target_col="all", treatment_col="all")
    df = bunch.data.copy()
    df["treatment"] = bunch.treatment["treatment"]
    df["exposure"] = bunch.treatment["exposure"]
    df["conversion"] = bunch.target["conversion"]
    df["visit"] = bunch.target["visit"]
    return df


def load_criteo_uplift(validate: bool = True) -> pd.DataFrame:
    try:
        logger.info("Attempting load from Hugging Face (criteo/criteo-uplift)")
        df = _load_from_huggingface()
        logger.info("Hugging Face load succeeded")
    except Exception as exc:
        logger.warning("Hugging Face load failed (%s), falling back to sklift", exc)
        df = _load_from_sklift()
        logger.info("sklift fallback load succeeded")

    if validate:
        run_integrity_check(df)

    return df


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    data = load_criteo_uplift()
    print(data.shape)
    print(data.head())
