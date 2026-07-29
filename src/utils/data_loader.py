import logging

import pandas as pd

logger = logging.getLogger(__name__)

EXPECTED_FEATURE_COLUMNS = [f"f{i}" for i in range(12)]
EXPECTED_COLUMNS = EXPECTED_FEATURE_COLUMNS + ["treatment", "exposure", "conversion", "visit"]

EXPECTED_ROW_COUNT_MIN = 13_000_000
EXPECTED_ROW_COUNT_MAX = 14_500_000

EXPECTED_TREATMENT_SHARE_MIN = 0.80
EXPECTED_TREATMENT_SHARE_MAX = 0.90
EXPECTED_VISIT_RATE_MIN = 0.03
EXPECTED_VISIT_RATE_MAX = 0.06
EXPECTED_CONVERSION_RATE_MIN = 0.001
EXPECTED_CONVERSION_RATE_MAX = 0.006


class DataIntegrityError(Exception):
    pass


def _check_columns(df: pd.DataFrame) -> None:
    missing = set(EXPECTED_COLUMNS) - set(df.columns)
    extra = set(df.columns) - set(EXPECTED_COLUMNS)
    if missing:
        raise DataIntegrityError(f"Missing expected columns: {sorted(missing)}")
    if extra:
        logger.warning("Unexpected extra columns found: %s", sorted(extra))


def _check_row_count(df: pd.DataFrame) -> None:
    n = len(df)
    if not (EXPECTED_ROW_COUNT_MIN <= n <= EXPECTED_ROW_COUNT_MAX):
        raise DataIntegrityError(
            f"Row count {n} outside expected band "
            f"[{EXPECTED_ROW_COUNT_MIN}, {EXPECTED_ROW_COUNT_MAX}]"
        )
    logger.info("Row count check passed: %d rows", n)


def _check_treatment_split(df: pd.DataFrame) -> None:
    share = df["treatment"].mean()
    if not (EXPECTED_TREATMENT_SHARE_MIN <= share <= EXPECTED_TREATMENT_SHARE_MAX):
        raise DataIntegrityError(
            f"Treatment share {share:.4f} outside expected band "
            f"[{EXPECTED_TREATMENT_SHARE_MIN}, {EXPECTED_TREATMENT_SHARE_MAX}]"
        )
    logger.info("Treatment split check passed: %.4f treated", share)


def _check_outcome_rates(df: pd.DataFrame) -> None:
    visit_rate = df["visit"].mean()
    conversion_rate = df["conversion"].mean()

    if not (EXPECTED_VISIT_RATE_MIN <= visit_rate <= EXPECTED_VISIT_RATE_MAX):
        raise DataIntegrityError(
            f"Visit rate {visit_rate:.4f} outside expected band "
            f"[{EXPECTED_VISIT_RATE_MIN}, {EXPECTED_VISIT_RATE_MAX}]"
        )
    if not (EXPECTED_CONVERSION_RATE_MIN <= conversion_rate <= EXPECTED_CONVERSION_RATE_MAX):
        raise DataIntegrityError(
            f"Conversion rate {conversion_rate:.4f} outside expected band "
            f"[{EXPECTED_CONVERSION_RATE_MIN}, {EXPECTED_CONVERSION_RATE_MAX}]"
        )
    logger.info(
        "Outcome rate check passed: visit=%.4f, conversion=%.4f",
        visit_rate,
        conversion_rate,
    )


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