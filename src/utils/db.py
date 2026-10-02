import json
import logging
import os
import sqlite3
from contextlib import closing
from datetime import datetime, timezone

import pandas as pd

logger = logging.getLogger(__name__)

_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
LOCAL_DB_PATH = os.path.join(_PROJECT_ROOT, "data", "local_runs.db")
TABLE_NAME = "estimation_runs"

_CREATE_TABLE_SQL = f"""
CREATE TABLE IF NOT EXISTS {TABLE_NAME} (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    method TEXT NOT NULL,
    severity_label TEXT NOT NULL,
    g2 REAL,
    config TEXT,
    point_estimate REAL,
    ci_lower REAL,
    ci_upper REAL,
    balance_stats TEXT,
    created_at TEXT NOT NULL,
    UNIQUE(method, severity_label)
)
"""


def get_supabase_client(url: str = None, key: str = None):
    url = url or os.environ.get("SUPABASE_URL")
    key = key or os.environ.get("SUPABASE_KEY")

    if not url or not key:
        logger.info("Supabase credentials not configured, will use local SQLite fallback")
        return None

    try:
        from supabase import create_client
        return create_client(url, key)
    except Exception as exc:
        logger.warning("Supabase client init failed (%s), falling back to local SQLite", exc)
        return None


def _resolve_path(path: str = None) -> str:
    return path if path is not None else LOCAL_DB_PATH


def _init_local_store(path: str = None) -> None:
    path = _resolve_path(path)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with closing(sqlite3.connect(path)) as conn:
        conn.execute(_CREATE_TABLE_SQL)
        conn.commit()


def _log_run_local(
    method: str,
    severity_label: str,
    g2: float,
    config: dict,
    point_estimate: float,
    ci_lower: float,
    ci_upper: float,
    balance_stats: dict,
    path: str = None,
) -> None:
    path = _resolve_path(path)
    _init_local_store(path)
    with closing(sqlite3.connect(path)) as conn:
        conn.execute(
            f"""
            INSERT INTO {TABLE_NAME}
            (method, severity_label, g2, config, point_estimate, ci_lower, ci_upper, balance_stats, created_at)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(method, severity_label) DO UPDATE SET
                g2=excluded.g2,
                config=excluded.config,
                point_estimate=excluded.point_estimate,
                ci_lower=excluded.ci_lower,
                ci_upper=excluded.ci_upper,
                balance_stats=excluded.balance_stats,
                created_at=excluded.created_at
            """,
            (
                method,
                severity_label,
                g2,
                json.dumps(config or {}),
                point_estimate,
                ci_lower,
                ci_upper,
                json.dumps(balance_stats or {}),
                datetime.now(timezone.utc).isoformat(),
            ),
        )
        conn.commit()


def _fetch_runs_local(path: str = None, method: str = None, severity_label: str = None) -> pd.DataFrame:
    path = _resolve_path(path)
    if not os.path.exists(path):
        return pd.DataFrame()

    query = f"SELECT * FROM {TABLE_NAME}"
    conditions = []
    params = []

    if method is not None:
        conditions.append("method = ?")
        params.append(method)
    if severity_label is not None:
        conditions.append("severity_label = ?")
        params.append(severity_label)

    if conditions:
        query += " WHERE " + " AND ".join(conditions)

    with closing(sqlite3.connect(path)) as conn:
        return pd.read_sql_query(query, conn, params=params)


def log_estimation_run(
    method: str,
    severity_label: str,
    g2: float,
    point_estimate: float,
    ci_lower: float,
    ci_upper: float,
    config: dict = None,
    balance_stats: dict = None,
    client=None,
) -> None:
    if not isinstance(severity_label, str) or not severity_label:
        raise ValueError("severity_label must be a non-empty string")

    if client is None:
        client = get_supabase_client()

    if client is not None:
        try:
            client.table(TABLE_NAME).upsert(
                {
                    "method": method,
                    "severity_label": severity_label,
                    "g2": g2,
                    "config": json.dumps(config or {}),
                    "point_estimate": point_estimate,
                    "ci_lower": ci_lower,
                    "ci_upper": ci_upper,
                    "balance_stats": json.dumps(balance_stats or {}),
                    "created_at": datetime.now(timezone.utc).isoformat(),
                },
                on_conflict="method,severity_label",
            ).execute()
            logger.info("Logged run (%s, %s) to Supabase", method, severity_label)
            return
        except Exception as exc:
            logger.warning("Supabase insert failed (%s), falling back to local SQLite", exc)

    _log_run_local(method, severity_label, g2, config, point_estimate, ci_lower, ci_upper, balance_stats)
    logger.info("Logged run (%s, %s) to local SQLite", method, severity_label)


def _parse_json_columns(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    for col in ("config", "balance_stats"):
        if col in df.columns:
            df[col] = df[col].apply(lambda s: json.loads(s) if isinstance(s, str) and s else (s or {}))
    return df


def _dedupe_latest(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty or "created_at" not in df.columns:
        return df
    df = df.sort_values("created_at")
    return df.drop_duplicates(subset=["method", "severity_label"], keep="last").reset_index(drop=True)


def fetch_estimation_runs(method: str = None, severity_label: str = None, client=None) -> pd.DataFrame:
    if client is None:
        client = get_supabase_client()

    frames = []

    if client is not None:
        try:
            query = client.table(TABLE_NAME).select("*")
            if method is not None:
                query = query.eq("method", method)
            if severity_label is not None:
                query = query.eq("severity_label", severity_label)
            remote = pd.DataFrame(query.execute().data)
            logger.info("Fetched %d runs from Supabase", len(remote))
            if not remote.empty:
                frames.append(remote)
        except Exception as exc:
            logger.warning("Supabase fetch failed (%s), using local SQLite only", exc)

    local = _fetch_runs_local(method=method, severity_label=severity_label)
    if not local.empty:
        frames.append(local)

    if not frames:
        return pd.DataFrame()

    combined = pd.concat([_parse_json_columns(f) for f in frames], ignore_index=True)
    return _dedupe_latest(combined)
