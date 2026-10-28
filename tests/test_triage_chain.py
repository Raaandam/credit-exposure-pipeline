"""Triage chain with a mocked structured LLM: happy path, fallbacks, masking, audit + HITL."""
from __future__ import annotations

from datetime import date

import pandas as pd
import pytest

pytest.importorskip("langchain_core")
from langchain_core.messages import AIMessage  # noqa: E402

from exposure_pipeline.dq.engine import DQResult  # noqa: E402
from exposure_pipeline.dq.results import write_results  # noqa: E402
from exposure_pipeline.storage.db import ensure_schema, get_engine, read_sql  # noqa: E402
from exposure_pipeline.triage.audit import AuditLog, ReviewError  # noqa: E402
from exposure_pipeline.triage.chain import TriageChain  # noqa: E402
from exposure_pipeline.triage.guardrails import Masker, TokenBudget  # noqa: E402
from exposure_pipeline.triage.schemas import NEEDS_MANUAL_REVIEW, TriageSuggestion  # noqa: E402

CONTEXT = {
    "break": {"business_date": "2026-01-07", "rule_id": "FAC_REF_001", "feed": "facilities", "failed_count": 2,
              "message": "2 rows violate foreign_key"},
    "rule_definition": {"rule_id": "FAC_REF_001", "name": "Facility counterparty exists in counterparty master"},
    "sample_failing_records": [{"facility_id": "FAC0000004", "counterparty_id": "CP999999", "counterparty_name": "Orion Pharma Inc"}],
    "failing_keys_sample": ["FAC0000004"],
    "feed_metadata": {"source_system": "LOAN_BOOKING"},
    "last_successful_load": {"business_date": "2026-01-06"},
    "other_breaks_same_day": [],
}
GOOD = {"category": "mapping_error", "confidence": 0.86,
        "root_cause_note": "The facility references a counterparty id absent from today's master.",
        "suggested_fix": "Ask loan booking to correct the counterparty id on the facility."}


class StubLLM:
    """Mimics ChatOpenAI.with_structured_output(..., include_raw=True)."""

    def __init__(self, parsed=None, error: Exception | None = None, parsing_error: Exception | None = None, tool_args=None):
        self.parsed, self.error, self.parsing_error, self.tool_args = parsed, error, parsing_error, tool_args
        self.calls: list = []

    def invoke(self, messages):
        self.calls.append(messages)
        if self.error:
            raise self.error
        raw = AIMessage(content="", tool_calls=[{"name": "TriageSuggestion", "args": self.tool_args or GOOD, "id": "call_1"}],
                        response_metadata={"model_name": "gpt-4o-mini-2024-07-18"},
                        usage_metadata={"input_tokens": 812, "output_tokens": 95, "total_tokens": 907})
        return {"raw": raw, "parsed": self.parsed, "parsing_error": self.parsing_error}


def make_chain(llm, threshold=0.7, budget=4000, prices=(0.15, 0.60)):
    return TriageChain(llm, "gpt-4o-mini", Masker("test-salt"), TokenBudget(budget), threshold, *prices)


def test_happy_path_returns_structured_suggestion():
    llm = StubLLM(parsed=TriageSuggestion(**GOOD))
    out = make_chain(llm).triage(CONTEXT)
    assert out.suggested_category == "mapping_error" and out.fallback_reason is None
    assert out.confidence == pytest.approx(0.86)
    assert out.model_version == "gpt-4o-mini-2024-07-18"
    assert out.input_tokens == 812 and out.output_tokens == 95
    assert out.cost_usd == pytest.approx(812 / 1e6 * 0.15 + 95 / 1e6 * 0.60)
    assert len(out.prompt_hash) == 64 and "mapping_error" in out.output_text


def test_prompt_sent_is_masked():
    llm = StubLLM(parsed=TriageSuggestion(**GOOD))
    out = make_chain(llm).triage(CONTEXT)
    sent = "\n".join(m.content for m in llm.calls[0])
    assert "CP999999" not in sent and "Orion Pharma" not in sent
    assert "FAC0000004" in sent  # facility ids are not personal/counterparty data
    assert out.prompt_text.endswith(llm.calls[0][-1].content)


def test_prompt_hash_is_stable_for_same_input():
    a = make_chain(StubLLM(parsed=TriageSuggestion(**GOOD))).triage(CONTEXT)
    b = make_chain(StubLLM(parsed=TriageSuggestion(**GOOD))).triage(CONTEXT)
    assert a.prompt_hash == b.prompt_hash


def test_low_confidence_falls_back_but_keeps_model_guess():
    out = make_chain(StubLLM(parsed=TriageSuggestion(**{**GOOD, "confidence": 0.41}))).triage(CONTEXT)
    assert out.suggested_category == NEEDS_MANUAL_REVIEW
    assert out.llm_category == "mapping_error" and out.fallback_reason.startswith("low_confidence")


def test_llm_exception_falls_back():
    out = make_chain(StubLLM(error=TimeoutError("read timeout"))).triage(CONTEXT)
    assert out.suggested_category == NEEDS_MANUAL_REVIEW and out.fallback_reason == "llm_error: TimeoutError"


def test_unparseable_output_falls_back():
    llm = StubLLM(parsed=None, parsing_error=ValueError("bad json"), tool_args={**GOOD, "category": "fraud"})
    out = make_chain(llm).triage(CONTEXT)
    assert out.suggested_category == NEEDS_MANUAL_REVIEW and out.fallback_reason.startswith("schema_validation_failed")
    assert out.llm_category is None


def test_recovers_valid_tool_args_when_langchain_parse_failed():
    out = make_chain(StubLLM(parsed=None, parsing_error=ValueError("x"))).triage(CONTEXT)
    assert out.suggested_category == "mapping_error"


def test_budget_exceeded_makes_no_call():
    llm = StubLLM(parsed=TriageSuggestion(**GOOD))
    out = make_chain(llm, budget=10).triage(CONTEXT)
    assert out.suggested_category == NEEDS_MANUAL_REVIEW and out.fallback_reason.startswith("token_budget_exceeded")
    assert llm.calls == []


def test_leak_blocks_call(monkeypatch):
    llm = StubLLM(parsed=TriageSuggestion(**GOOD))
    chain = make_chain(llm)
    monkeypatch.setattr(chain.masker, "mask", lambda ctx: (ctx, {"CP999999"}))  # simulate a masking bug
    out = chain.triage(CONTEXT)
    assert out.fallback_reason.startswith("pii_leak_blocked") and llm.calls == []
    assert "CP999999" not in out.prompt_text


# ----------------------------------------------------------------------------- audit + HITL
@pytest.fixture
def db(tmp_path):
    engine = get_engine(f"sqlite:///{(tmp_path / 'audit.db').as_posix()}")
    ensure_schema(engine)
    bd = date(2026, 1, 7)
    res = DQResult(result_id="2026-01-07:FAC_REF_001", run_id="r1", business_date=bd, rule_id="FAC_REF_001",
                   rule_name="x", feed="facilities", dataset="raw.facilities", dimension="referential_integrity",
                   check_type="foreign_key", severity="high", status="FAIL", failed_count=2, total_count=10)
    write_results(engine, [res], bd)
    return engine


def _dq_row(engine):
    return read_sql(engine, "SELECT triage_status, triage_category, triage_note FROM dq_results").iloc[0]


def test_suggestion_is_not_written_back_until_approved(db):
    audit = AuditLog(db)
    out = make_chain(StubLLM(parsed=TriageSuggestion(**GOOD))).triage(CONTEXT)
    aid = audit.record_suggestion(out, "2026-01-07:FAC_REF_001", "2026-01-07", "FAC_REF_001")
    row = _dq_row(db)
    assert row["triage_status"] == "untriaged" and row["triage_category"] is None
    pending = audit.pending()
    assert list(pending["audit_id"]) == [aid]
    stored = pending.iloc[0]
    assert stored["prompt_text"] == out.prompt_text and stored["model_version"] == "gpt-4o-mini-2024-07-18"
    assert stored["prompt_hash"] == out.prompt_hash and stored["output_text"] == out.output_text

    audit.record_review(aid, "analyst.one", "approved")
    row = _dq_row(db)
    assert row["triage_status"] == "approved" and row["triage_category"] == "mapping_error"
    assert audit.pending().empty


def test_review_is_write_once(db):
    audit = AuditLog(db)
    aid = audit.record_suggestion(make_chain(StubLLM(parsed=TriageSuggestion(**GOOD))).triage(CONTEXT),
                                  "2026-01-07:FAC_REF_001", "2026-01-07", "FAC_REF_001")
    audit.record_review(aid, "a", "edited", final_category="missing_source_data", final_note="Master truncated.")
    with pytest.raises(ReviewError, match="already reviewed"):
        audit.record_review(aid, "b", "approved")
    hist = audit.history("2026-01-07:FAC_REF_001").iloc[0]
    assert hist["reviewer_decision"] == "edited" and hist["final_category"] == "missing_source_data"
    assert hist["llm_category"] == "mapping_error"  # model output preserved next to the human decision


def test_fallback_cannot_be_blindly_approved(db):
    audit = AuditLog(db)
    out = make_chain(StubLLM(error=RuntimeError("down"))).triage(CONTEXT)
    aid = audit.record_suggestion(out, "2026-01-07:FAC_REF_001", "2026-01-07", "FAC_REF_001")
    with pytest.raises(ReviewError):
        audit.record_review(aid, "a", "approved")
    with pytest.raises(ReviewError, match="reviewer"):
        audit.record_review(aid, " ", "edited", final_category="duplicate")
    audit.record_review(aid, "a", "rejected", final_category="duplicate")
    assert _dq_row(db)["triage_status"] == "rejected"


def test_runner_end_to_end_with_stub(db, monkeypatch):
    from exposure_pipeline.triage import runner

    n = runner.run_triage(engine=db, chain=make_chain(StubLLM(parsed=TriageSuggestion(**GOOD))))
    assert n == 1
    # second run does not re-bill a break that already has a pending suggestion
    assert runner.run_triage(engine=db, chain=make_chain(StubLLM(parsed=TriageSuggestion(**GOOD)))) == 0
    assert isinstance(AuditLog(db).pending(), pd.DataFrame)
