"""Audit log + human-in-the-loop write-back.

Two invariants (both unit-tested):
1. Recording an LLM suggestion writes ONLY to ``triage_audit``. ``dq_results`` - what the
   dashboard and downstream consumers read - is untouched until a human decides.
2. A review is written exactly once (``UPDATE ... WHERE reviewer_decision IS NULL``) in
   the same transaction as the dq_results write-back, so the two cannot disagree. In
   Postgres a trigger additionally makes model-output columns immutable.
"""
from __future__ import annotations

import uuid
from datetime import date, datetime, timezone

import pandas as pd
from sqlalchemy import and_, insert, select, update
from sqlalchemy.engine import Engine

from exposure_pipeline.storage.tables import dq_results, triage_audit
from exposure_pipeline.triage.chain import TriageOutcome
from exposure_pipeline.triage.schemas import ALL_CATEGORIES, NEEDS_MANUAL_REVIEW

DECISIONS = {"approved", "edited", "rejected"}


class ReviewError(ValueError):
    pass


def _now() -> datetime:
    return datetime.now(timezone.utc).replace(tzinfo=None)


class AuditLog:
    def __init__(self, engine: Engine):
        self.engine = engine

    def record_suggestion(self, outcome: TriageOutcome, result_id: str, business_date: date | str | None, rule_id: str | None) -> str:
        audit_id = str(uuid.uuid4())
        if isinstance(business_date, str):
            business_date = date.fromisoformat(business_date)
        row = {
            "audit_id": audit_id, "result_id": result_id, "business_date": business_date, "rule_id": rule_id,
            "created_at": _now(), "model_requested": outcome.model_requested, "model_version": outcome.model_version,
            "prompt_version": outcome.prompt_version, "temperature": outcome.temperature,
            "prompt_hash": outcome.prompt_hash, "prompt_text": outcome.prompt_text, "output_text": outcome.output_text,
            "llm_category": outcome.llm_category, "suggested_category": outcome.suggested_category,
            "confidence": outcome.confidence, "root_cause_note": outcome.root_cause_note,
            "suggested_fix": outcome.suggested_fix, "fallback_reason": outcome.fallback_reason,
            "input_tokens": outcome.input_tokens, "output_tokens": outcome.output_tokens,
            "latency_ms": outcome.latency_ms, "cost_usd": outcome.cost_usd,
        }
        with self.engine.begin() as conn:
            conn.execute(insert(triage_audit), [row])
        return audit_id

    def pending(self, business_date: date | None = None) -> pd.DataFrame:
        stmt = select(triage_audit).where(triage_audit.c.reviewer_decision.is_(None)).order_by(
            triage_audit.c.business_date, triage_audit.c.rule_id)
        if business_date:
            stmt = stmt.where(triage_audit.c.business_date == business_date)
        with self.engine.connect() as conn:
            return pd.DataFrame(conn.execute(stmt).mappings().all())

    def history(self, result_id: str) -> pd.DataFrame:
        stmt = select(triage_audit).where(triage_audit.c.result_id == result_id).order_by(triage_audit.c.created_at)
        with self.engine.connect() as conn:
            return pd.DataFrame(conn.execute(stmt).mappings().all())

    def record_review(
        self,
        audit_id: str,
        reviewer: str,
        decision: str,
        final_category: str | None = None,
        final_note: str | None = None,
    ) -> None:
        """Apply a human decision.

        * approved - accept the suggestion as is (not allowed for needs_manual_review: the
          reviewer must pick a real category, i.e. use 'edited').
        * edited   - accept with reviewer-chosen category and/or note.
        * rejected - discard the suggestion; dq_results gets status 'rejected' and, if the
          reviewer supplies one, the correct category (useful as an analyst label for eval).
        """
        if decision not in DECISIONS:
            raise ReviewError(f"decision must be one of {sorted(DECISIONS)}")
        if not reviewer or not reviewer.strip():
            raise ReviewError("reviewer is required")
        with self.engine.begin() as conn:
            row = conn.execute(select(triage_audit).where(triage_audit.c.audit_id == audit_id)).mappings().first()
            if row is None:
                raise ReviewError(f"unknown audit_id {audit_id}")
            if decision == "approved":
                if row["suggested_category"] == NEEDS_MANUAL_REVIEW:
                    raise ReviewError("cannot approve needs_manual_review; choose a category (edited)")
                final_category = row["suggested_category"]
                final_note = row["root_cause_note"]
            elif decision == "edited":
                final_category = final_category or row["suggested_category"]
                final_note = final_note if final_note is not None else row["root_cause_note"]
            if final_category is not None and final_category not in ALL_CATEGORIES:
                raise ReviewError(f"unknown category {final_category}")
            if decision == "edited" and final_category == NEEDS_MANUAL_REVIEW:
                raise ReviewError("edited reviews must set a real category")

            now = _now()
            res = conn.execute(
                update(triage_audit)
                .where(and_(triage_audit.c.audit_id == audit_id, triage_audit.c.reviewer_decision.is_(None)))
                .values(reviewer=reviewer.strip(), reviewer_decision=decision, reviewed_at=now,
                        final_category=final_category, final_note=final_note)
            )
            if res.rowcount != 1:
                raise ReviewError(f"audit {audit_id} was already reviewed")

            values = {"triage_reviewed_by": reviewer.strip(), "triage_reviewed_at": now}
            if decision == "rejected":
                values.update(triage_status="rejected", triage_category=final_category, triage_note=final_note)
            else:
                values.update(triage_status="approved", triage_category=final_category, triage_note=final_note)
            conn.execute(update(dq_results).where(dq_results.c.result_id == row["result_id"]).values(**values))
