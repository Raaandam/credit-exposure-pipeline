"""Portable SQLAlchemy tables must stay in sync with the authoritative Postgres DDL."""
from __future__ import annotations

import re
from pathlib import Path

import pytest

from exposure_pipeline.storage.tables import metadata

SQL_DIR = Path(__file__).resolve().parents[1] / "sql"


def _ddl_columns() -> dict[str, set[str]]:
    text = "\n".join(p.read_text() for p in sorted(SQL_DIR.glob("*.sql")))
    tables = {}
    for name, body in re.findall(r"CREATE TABLE IF NOT EXISTS (\w+) \((.*?)\n\);", text, re.DOTALL):
        cols = set()
        for line in body.splitlines():
            line = line.strip()
            if line and not line.startswith(("PRIMARY KEY", "--")):
                cols.add(line.split()[0])
        tables[name] = cols
    return tables


@pytest.mark.parametrize("table", sorted(metadata.tables))
def test_tables_match_ddl(table):
    ddl = _ddl_columns()
    assert table in ddl, f"{table} missing from sql/*.sql"
    assert set(metadata.tables[table].columns.keys()) == ddl[table]
