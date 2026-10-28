"""Batch triage: generate suggestions for untriaged breaks (suggestions only - no write-back).

    python -m exposure_pipeline.triage.runner --business-date 2026-01-07 --limit 20
"""
from __future__ import annotations

import argparse
import logging
from datetime import date

from sqlalchemy.engine import Engine

from exposure_pipeline.config import get_settings
from exposure_pipeline.dq.rules import load_rules
from exposure_pipeline.storage.db import ensure_schema, get_engine, read_sql
from exposure_pipeline.triage.audit import AuditLog
from exposure_pipeline.triage.chain import TriageChain
from exposure_pipeline.triage.context import fetch_context

log = logging.getLogger(__name__)


def untriaged_breaks(engine: Engine, business_date: date | None = None, limit: int | None = None) -> list[dict]:
    """Breaks with no decision yet and no pending suggestion (so reruns don't re-bill)."""
    sql = """
        SELECT r.result_id, r.business_date, r.rule_id FROM dq_results r
        WHERE r.status IN ('FAIL', 'ERROR') AND COALESCE(r.triage_status, 'untriaged') = 'untriaged'
          AND NOT EXISTS (SELECT 1 FROM triage_audit a WHERE a.result_id = r.result_id AND a.reviewer_decision IS NULL)
    """
    params = {}
    if business_date:
        sql += " AND r.business_date = :bd"
        params["bd"] = business_date.isoformat()
    sql += " ORDER BY r.business_date, r.rule_id"
    if limit:
        sql += f" LIMIT {int(limit)}"
    return read_sql(engine, sql, **params).to_dict("records")


def run_triage(business_date: date | None = None, limit: int | None = None, engine: Engine | None = None,
               chain: TriageChain | None = None) -> int:
    settings = get_settings()
    engine = engine or get_engine(settings.database_url)
    ensure_schema(engine)
    chain = chain or TriageChain.from_settings(settings)
    audit = AuditLog(engine)
    rules = {r.rule_id: r for r in load_rules()}
    n = 0
    for brk in untriaged_breaks(engine, business_date, limit):
        context = fetch_context(engine, brk["result_id"], rules)
        outcome = chain.triage(context)
        audit.record_suggestion(outcome, brk["result_id"], str(brk["business_date"])[:10], brk["rule_id"])
        log.info("%s -> %s (conf=%s, fallback=%s)", brk["result_id"], outcome.suggested_category,
                 outcome.confidence, outcome.fallback_reason)
        n += 1
    return n


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description="Generate LLM triage suggestions for DQ breaks")
    ap.add_argument("--business-date", type=date.fromisoformat)
    ap.add_argument("--limit", type=int)
    args = ap.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    print(f"suggestions recorded: {run_triage(args.business_date, args.limit)}")


if __name__ == "__main__":
    main()
