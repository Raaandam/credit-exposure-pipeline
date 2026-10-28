"""Database helpers: engine creation and idempotent partition writes.

Idempotency pattern: every write for a business date is "delete partition, insert
partition" inside ONE transaction. Re-running a day (Airflow retry, backfill, manual
rerun) therefore converges to the same state and never double-counts exposure.
"""
from __future__ import annotations

import json
import math
from datetime import date, datetime
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pandas as pd
from sqlalchemy import Table, create_engine, delete, insert, text
from sqlalchemy.engine import Connection, Engine

from exposure_pipeline.config import get_settings
from exposure_pipeline.storage.tables import metadata


def get_engine(url: str | None = None) -> Engine:
    url = url or get_settings().database_url
    if url.startswith("sqlite:///") and not url.startswith("sqlite:///:memory:"):
        Path(url.replace("sqlite:///", "", 1)).parent.mkdir(parents=True, exist_ok=True)
    return create_engine(url, future=True)


def ensure_schema(engine: Engine) -> None:
    """Create missing tables (no-op for tables created from sql/*.sql in Postgres)."""
    metadata.create_all(engine, checkfirst=True)


def to_python(value: Any) -> Any:
    """Convert numpy / pandas scalars to JSON- and DB-API-friendly Python values."""
    if value is None:
        return None
    if isinstance(value, np.generic):
        value = value.item()
    if isinstance(value, float) and math.isnan(value):
        return None
    if value is pd.NaT or (not isinstance(value, (list, dict, str)) and pd.isna(value)):
        return None
    if isinstance(value, pd.Timestamp):
        return value.to_pydatetime()
    return value


def json_default(value: Any) -> Any:
    v = to_python(value)
    if isinstance(v, (date, datetime)):
        return v.isoformat()
    return v if not isinstance(v, (np.generic,)) else str(v)


def dumps(obj: Any) -> str:
    return json.dumps(obj, default=json_default)


def records(df: pd.DataFrame, table: Table) -> list[dict]:
    cols = [c for c in table.columns.keys() if c in df.columns]
    return [{c: to_python(v) for c, v in zip(cols, row)} for row in df[cols].itertuples(index=False, name=None)]


def replace_partition(conn: Connection, table: Table, df: pd.DataFrame, business_date: date, **filters: Any) -> int:
    stmt = delete(table).where(table.c.business_date == business_date)
    for col, val in filters.items():
        stmt = stmt.where(table.c[col] == val)
    conn.execute(stmt)
    rows = records(df, table)
    if rows:
        conn.execute(insert(table), rows)
    return len(rows)


def swap_partition_from_staging(engine: Engine, target: str, staging: str, business_date: date, columns: Iterable[str]) -> None:
    """Used by the Spark job: Spark JDBC writes the day's rows to a staging table (fast,
    parallel, mode=overwrite), then this moves them into the target partition atomically.
    Spark's JDBC sink cannot do "delete where business_date = X" itself."""
    cols = ", ".join(columns)
    with engine.begin() as conn:
        conn.execute(text(f"DELETE FROM {target} WHERE business_date = :bd"), {"bd": business_date})
        conn.execute(text(f"INSERT INTO {target} ({cols}) SELECT {cols} FROM {staging}"))
        conn.execute(text(f"DROP TABLE IF EXISTS {staging}"))


def read_sql(engine: Engine, sql: str, **params: Any) -> pd.DataFrame:
    with engine.connect() as conn:
        return pd.read_sql(text(sql), conn, params=params)
