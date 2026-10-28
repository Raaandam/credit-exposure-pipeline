"""SQLAlchemy table definitions.

``sql/*.sql`` is the authoritative Postgres DDL (applied by the postgres container on first
start, with JSONB / indexes / constraints). These portable definitions mirror it so that the
same code can create the schema in SQLite for laptop runs and tests. If you change one,
change the other - ``tests/test_storage.py`` checks the column sets match.
"""
from __future__ import annotations

from sqlalchemy import Boolean, Column, Date, DateTime, Float, Integer, MetaData, String, Table, Text

metadata = MetaData()

cur_counterparty = Table(
    "cur_counterparty", metadata,
    Column("business_date", Date, nullable=False, index=True),
    Column("counterparty_id", String(32), nullable=False),
    Column("counterparty_name", String(200)),
    Column("lei", String(20)),
    Column("group_id", String(32)),
    Column("ultimate_parent_id", String(32)),
    Column("country", String(32)),
    Column("sector", String(64)),
    Column("internal_rating", String(8)),
    Column("rating_band", String(16)),
)

cur_facility_exposure = Table(
    "cur_facility_exposure", metadata,
    Column("business_date", Date, nullable=False, index=True),
    Column("facility_id", String(32), nullable=False),
    Column("counterparty_id", String(32)),
    Column("group_id", String(32)),
    Column("sector", String(64)),
    Column("rating_band", String(16)),
    Column("facility_type", String(32)),
    Column("currency", String(8)),
    Column("source_system", String(32)),
    Column("limit_amount", Float),
    Column("drawn_amount", Float),
    Column("fx_rate_to_chf", Float),
    Column("limit_chf", Float),
    Column("drawn_chf", Float),
    Column("undrawn_chf", Float),
    Column("ccf", Float),
    Column("ead_chf", Float),
    Column("collateral_chf", Float),
    Column("net_exposure_chf", Float),
    Column("maturity_date", String(10)),
    Column("is_orphan", Boolean),
)

cur_collateral = Table(
    "cur_collateral", metadata,
    Column("business_date", Date, nullable=False, index=True),
    Column("collateral_id", String(32), nullable=False),
    Column("facility_id", String(32)),
    Column("collateral_type", String(32)),
    Column("currency", String(8)),
    Column("market_value", Float),
    Column("haircut", Float),
    Column("adjusted_value_chf", Float),
)

cur_rating_change = Table(
    "cur_rating_change", metadata,
    Column("business_date", Date, nullable=False, index=True),
    Column("counterparty_id", String(32), nullable=False),
    Column("previous_rating", String(8)),
    Column("new_rating", String(8)),
    Column("notch_change", Float),
    Column("notch_change_abs", Float),
    Column("effective_date", String(10)),
    Column("change_reason", String(64)),
)

mart_exposure_aggregate = Table(
    "mart_exposure_aggregate", metadata,
    Column("business_date", Date, nullable=False, index=True),
    Column("dimension", String(32), nullable=False),
    Column("dimension_value", String(64), nullable=False),
    Column("ead_chf", Float),
    Column("collateral_chf", Float),
    Column("net_exposure_chf", Float),
    Column("facility_count", Integer),
    Column("counterparty_count", Integer),
)

load_audit = Table(
    "load_audit", metadata,
    Column("business_date", Date, nullable=False, index=True),
    Column("feed", String(32), nullable=False),
    Column("run_id", String(64), nullable=False),
    Column("status", String(16), nullable=False),  # SUCCESS | MISSING | FAILED
    Column("row_count", Integer),
    Column("source_path", Text),
    Column("source_system", String(64)),
    Column("manifest_business_date", String(10)),
    Column("manifest_record_count", Integer),
    Column("extract_timestamp", String(32)),
    Column("loaded_at", DateTime, nullable=False),
)

dq_results = Table(
    "dq_results", metadata,
    # Deterministic id "<business_date>:<rule_id>" -> reruns replace rather than duplicate,
    # and triage_audit rows keep pointing at the same logical break.
    Column("result_id", String(80), primary_key=True),
    Column("run_id", String(64), nullable=False),
    Column("business_date", Date, nullable=False, index=True),
    Column("rule_id", String(40), nullable=False),
    Column("rule_name", String(200)),
    Column("feed", String(32)),
    Column("dataset", String(64)),
    Column("dimension", String(32)),
    Column("check_type", String(40)),
    Column("severity", String(16)),
    Column("status", String(8), nullable=False),  # PASS | FAIL | ERROR
    Column("total_count", Integer),
    Column("failed_count", Integer),
    Column("failure_rate", Float),
    Column("observed_value", Text),
    Column("expected_value", Text),
    Column("message", Text),
    Column("failed_keys", Text),  # JSON list (JSONB in Postgres)
    Column("sample_records", Text),  # JSON list of records (JSONB in Postgres)
    Column("executed_at", DateTime, nullable=False),
    # Written ONLY by an approved human review (see triage.audit.AuditLog.record_review)
    Column("triage_status", String(16)),  # n/a | untriaged | approved | rejected
    Column("triage_category", String(40)),
    Column("triage_note", Text),
    Column("triage_reviewed_by", String(100)),
    Column("triage_reviewed_at", DateTime),
)

triage_audit = Table(
    "triage_audit", metadata,
    Column("audit_id", String(36), primary_key=True),
    Column("result_id", String(80), nullable=False, index=True),
    Column("business_date", Date),
    Column("rule_id", String(40)),
    Column("created_at", DateTime, nullable=False),
    Column("model_requested", String(64)),
    Column("model_version", String(128)),
    Column("prompt_version", String(32)),
    Column("temperature", Float),
    Column("prompt_hash", String(64)),
    Column("prompt_text", Text),
    Column("output_text", Text),
    Column("llm_category", String(40)),
    Column("suggested_category", String(40)),
    Column("confidence", Float),
    Column("root_cause_note", Text),
    Column("suggested_fix", Text),
    Column("fallback_reason", Text),
    Column("input_tokens", Integer),
    Column("output_tokens", Integer),
    Column("latency_ms", Float),
    Column("cost_usd", Float),
    Column("reviewer", String(100)),
    Column("reviewer_decision", String(16)),  # approved | edited | rejected
    Column("reviewed_at", DateTime),
    Column("final_category", String(40)),
    Column("final_note", Text),
)

TABLES_BY_DATASET = {
    "curated.counterparty": cur_counterparty,
    "curated.facility_exposure": cur_facility_exposure,
    "curated.collateral": cur_collateral,
    "curated.rating_change": cur_rating_change,
    "mart.exposure_aggregate": mart_exposure_aggregate,
}
