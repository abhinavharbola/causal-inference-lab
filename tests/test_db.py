import pytest

from src.utils import db


@pytest.fixture
def local_store(tmp_path, monkeypatch):
    monkeypatch.setattr(db, "LOCAL_DB_PATH", str(tmp_path / "runs.db"))
    monkeypatch.delenv("SUPABASE_URL", raising=False)
    monkeypatch.delenv("SUPABASE_KEY", raising=False)
    return tmp_path


def _log(method="ipw", severity="mild", estimate=0.01):
    db.log_estimation_run(
        method=method,
        severity_label=severity,
        g2=0.5,
        point_estimate=estimate,
        ci_lower=estimate - 0.001,
        ci_upper=estimate + 0.001,
        config={"n": 1},
        balance_stats={"x": 1},
    )


def test_log_and_fetch_round_trip_locally(local_store):
    _log()

    runs = db.fetch_estimation_runs()

    assert len(runs) == 1
    assert runs.loc[0, "config"] == {"n": 1}
    assert runs.loc[0, "point_estimate"] == pytest.approx(0.01)


def test_repeated_logging_upserts_instead_of_duplicating(local_store):
    _log(estimate=0.01)
    _log(estimate=0.02)

    runs = db.fetch_estimation_runs()

    assert len(runs) == 1
    assert runs.loc[0, "point_estimate"] == pytest.approx(0.02)


def test_null_or_empty_severity_is_rejected(local_store):
    with pytest.raises(ValueError):
        db.log_estimation_run("ipw", None, 0.5, 0.01, 0.0, 0.02)
    with pytest.raises(ValueError):
        db.log_estimation_run("ipw", "", 0.5, 0.01, 0.0, 0.02)


def test_fetch_filters_by_method_and_severity(local_store):
    _log("ipw", "mild")
    _log("psm", "mild")
    _log("ipw", "strong")

    assert len(db.fetch_estimation_runs(method="ipw")) == 2
    assert len(db.fetch_estimation_runs(severity_label="mild")) == 2
    assert len(db.fetch_estimation_runs(method="psm", severity_label="mild")) == 1


class _FailingTable:
    def upsert(self, *args, **kwargs):
        return self

    def select(self, *args, **kwargs):
        return self

    def eq(self, *args, **kwargs):
        return self

    def execute(self):
        raise RuntimeError("remote down")


class _FailingClient:
    def table(self, name):
        return _FailingTable()


def test_failed_remote_write_is_still_readable_afterwards(local_store):
    db.log_estimation_run("ipw", "mild", 0.5, 0.01, 0.0, 0.02, client=_FailingClient())

    runs = db.fetch_estimation_runs(client=_FailingClient())

    assert len(runs) == 1
    assert runs.loc[0, "method"] == "ipw"


class _Response:
    def __init__(self, data):
        self.data = data


class _RemoteTable:
    def __init__(self, rows):
        self.rows = rows

    def select(self, *args, **kwargs):
        return self

    def eq(self, *args, **kwargs):
        return self

    def execute(self):
        return _Response(self.rows)


class _RemoteClient:
    def __init__(self, rows):
        self.rows = rows

    def table(self, name):
        return _RemoteTable(self.rows)


def test_fetch_merges_remote_and_local_keeping_the_newest_row(local_store):
    _log("ipw", "mild", estimate=0.05)
    remote_rows = [
        {
            "method": "ipw",
            "severity_label": "mild",
            "g2": 0.5,
            "config": "{}",
            "point_estimate": 0.01,
            "ci_lower": 0.0,
            "ci_upper": 0.02,
            "balance_stats": "{}",
            "created_at": "2000-01-01T00:00:00+00:00",
        },
        {
            "method": "psm",
            "severity_label": "mild",
            "g2": 0.5,
            "config": "{}",
            "point_estimate": 0.02,
            "ci_lower": 0.0,
            "ci_upper": 0.03,
            "balance_stats": "{}",
            "created_at": "2000-01-01T00:00:00+00:00",
        },
    ]

    runs = db.fetch_estimation_runs(client=_RemoteClient(remote_rows)).set_index("method")

    assert set(runs.index) == {"ipw", "psm"}
    assert runs.loc["ipw", "point_estimate"] == pytest.approx(0.05)
