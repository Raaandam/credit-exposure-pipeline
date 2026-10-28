"""Assemble the evidence an analyst would look at for one DQ break.

Context = the rule definition, the measured result, sample failing records, feed metadata
from the source control file (manifest), the last successful load of that feed, and the
other breaks on the same day. The last two matter most for root cause: e.g. a foreign-key
break on facilities *plus* a record-count break on counterparties the same day points to
a truncated master file (missing_source_data), not to bad ids (mapping_error).

Everything here is unmasked; masking happens in the chain right before rendering.
"""
from __future__ import annotations

import json
from typing import Any

import pandas as pd
from sqlalchemy.engine import Engine

from exposure_pipeline.dq.rules import Rule
from exposure_pipeline.storage.db import read_sql


def _loads(value: Any) -> Any:
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return []
    if isinstance(value, (list, dict)):
        return value
    return json.loads(value)


def build_context(
    dq_row: dict,
    rule: Rule | None,
    load_meta: dict | None = None,
    last_success: dict | None = None,
    related: list[dict] | None = None,
) -> dict:
    return {
        "break": {k: dq_row.get(k) for k in (
            "business_date", "rule_id", "rule_name", "feed", "dataset", "dimension", "check_type", "severity",
            "status", "total_count", "failed_count", "failure_rate", "observed_value", "expected_value", "message")},
        "rule_definition": rule.as_prompt_dict() if rule else None,
        "sample_failing_records": _loads(dq_row.get("sample_records")),
        "failing_keys_sample": _loads(dq_row.get("failed_keys")),
        "feed_metadata": load_meta or {},
        "last_successful_load": last_success or {},
        "other_breaks_same_day": related or [],
    }


def _clean(d: dict) -> dict:
    out = {}
    for k, v in d.items():
        if isinstance(v, float) and pd.isna(v):
            v = None
        elif hasattr(v, "isoformat"):
            v = v.isoformat()
        out[k] = v
    return out


def fetch_context(engine: Engine, result_id: str, rules_by_id: dict[str, Rule]) -> dict:
    rows = read_sql(engine, "SELECT * FROM dq_results WHERE result_id = :rid", rid=result_id)
    if rows.empty:
        raise KeyError(result_id)
    row = _clean(rows.iloc[0].to_dict())
    bd, feed = row["business_date"], row["feed"]

    meta = read_sql(engine, """
        SELECT status, row_count, source_system, manifest_business_date, manifest_record_count, extract_timestamp
        FROM load_audit WHERE business_date = :bd AND feed = :feed""", bd=bd, feed=feed)
    last = read_sql(engine, """
        SELECT business_date, row_count, manifest_business_date, loaded_at
        FROM load_audit WHERE feed = :feed AND business_date < :bd AND status = 'SUCCESS'
        ORDER BY business_date DESC LIMIT 1""", bd=bd, feed=feed)
    related = read_sql(engine, """
        SELECT rule_id, feed, dimension, check_type, failed_count FROM dq_results
        WHERE business_date = :bd AND status IN ('FAIL', 'ERROR') AND result_id <> :rid
        ORDER BY severity, rule_id LIMIT 15""", bd=bd, rid=result_id)

    return build_context(
        row,
        rules_by_id.get(row["rule_id"]),
        _clean(meta.iloc[0].to_dict()) if not meta.empty else None,
        _clean(last.iloc[0].to_dict()) if not last.empty else None,
        [_clean(r) for r in related.to_dict("records")],
    )
