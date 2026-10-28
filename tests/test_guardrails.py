"""Masking, token budget, schema validation and confidence gate."""
from __future__ import annotations

import json

import pytest

from exposure_pipeline.triage.guardrails import (
    NAME_TOKEN,
    Masker,
    PIILeakError,
    SchemaValidationError,
    TokenBudget,
    TokenBudgetExceeded,
    confidence_gate,
    validate_suggestion,
)
from exposure_pipeline.triage.schemas import NEEDS_MANUAL_REVIEW, TriageSuggestion

CONTEXT = {
    "break": {"rule_id": "FAC_REF_001", "rule_name": "Facility counterparty exists", "message": "orphan CP000123 found"},
    "rule_definition": {"name": "Facility counterparty exists in counterparty master", "parameters": {"column": "counterparty_id"}},
    "sample_failing_records": [
        {"facility_id": "FAC0000001", "counterparty_id": "CP000123", "counterparty_name": "Alpine Holdings AG",
         "group_id": "GRP00042", "lei": "ABCDEFGHIJKLMNOPQR12", "limit_amount": 1e6},
        {"facility_id": "FAC0000002", "counterparty_id": "CP000123", "counterparty_name": "Alpine Holdings AG"},
    ],
    "failing_keys_sample": ["CP000123|2026-01-07", "CP000456"],
    "other_breaks_same_day": [{"rule_id": "REC_CP_CNT_001"}],
}


@pytest.fixture
def masker():
    return Masker("unit-test-salt")


def test_ids_and_names_are_masked_everywhere(masker):
    masked, sensitive = masker.mask(CONTEXT)
    text = json.dumps(masked)
    for raw in ("CP000123", "CP000456", "Alpine Holdings AG", "GRP00042", "ABCDEFGHIJKLMNOPQR12"):
        assert raw not in text
    assert {"CP000123", "Alpine Holdings AG", "GRP00042"} <= sensitive
    rec = masked["sample_failing_records"][0]
    assert rec["counterparty_name"] == NAME_TOKEN
    assert rec["counterparty_id"].startswith("CP#") and rec["group_id"].startswith("GRP#") and rec["lei"].startswith("LEI#")
    # non-sensitive values and rule metadata survive
    assert rec["facility_id"] == "FAC0000001" and rec["limit_amount"] == 1e6
    assert masked["rule_definition"]["name"] == CONTEXT["rule_definition"]["name"]
    assert masked["rule_definition"]["parameters"]["column"] == "counterparty_id"


def test_masking_is_deterministic_and_salted(masker):
    a, _ = masker.mask(CONTEXT)
    b, _ = masker.mask(CONTEXT)
    assert a == b
    # same id -> same token, so the model can still see two rows share a counterparty
    assert a["sample_failing_records"][0]["counterparty_id"] == a["sample_failing_records"][1]["counterparty_id"]
    assert a["failing_keys_sample"][0].startswith(a["sample_failing_records"][0]["counterparty_id"])
    other, _ = Masker("other-salt").mask(CONTEXT)
    assert other["sample_failing_records"][0]["counterparty_id"] != a["sample_failing_records"][0]["counterparty_id"]


def test_original_is_not_mutated(masker):
    masker.mask(CONTEXT)
    assert CONTEXT["sample_failing_records"][0]["counterparty_id"] == "CP000123"


def test_leak_check(masker):
    _, sensitive = masker.mask(CONTEXT)
    Masker.assert_no_leak("all clean CP#abc", sensitive)
    with pytest.raises(PIILeakError) as exc:
        Masker.assert_no_leak("oops Alpine Holdings AG", sensitive)
    assert "Alpine" not in str(exc.value)  # never echo the leaked value


def test_empty_salt_rejected():
    with pytest.raises(ValueError):
        Masker("")


def test_token_budget_shrinks_context():
    big = {**CONTEXT, "sample_failing_records": [{"x": "y" * 400}] * 5, "failing_keys_sample": ["k"] * 20}
    render = lambda c: json.dumps(c)  # noqa: E731
    full = len(render(big)) // 4
    budget = TokenBudget(max_input_tokens=full // 2)
    fitted, text, n = budget.fit(big, render)
    assert n <= full // 2
    assert len(fitted["sample_failing_records"]) < 5
    assert len(big["sample_failing_records"]) == 5  # input untouched


def test_token_budget_exceeded():
    with pytest.raises(TokenBudgetExceeded):
        TokenBudget(max_input_tokens=5).fit(CONTEXT, lambda c: json.dumps(c))


VALID = {"category": "mapping_error", "confidence": 0.9, "root_cause_note": "Facility references an unknown counterparty.",
         "suggested_fix": "Ask loan booking to correct the counterparty id."}


@pytest.mark.parametrize("payload", [VALID, json.dumps(VALID), TriageSuggestion(**VALID)])
def test_validate_accepts_dict_json_and_model(payload):
    assert validate_suggestion(payload).category == "mapping_error"


@pytest.mark.parametrize("bad", [
    {**VALID, "category": "needs_manual_review"},     # reserved for guardrails
    {**VALID, "category": "fraud"},
    {**VALID, "confidence": 1.7},
    {**VALID, "root_cause_note": "short"},
    {**VALID, "extra_field": 1},
    {k: v for k, v in VALID.items() if k != "suggested_fix"},
    "not json",
    42,
])
def test_validate_rejects_bad_output(bad):
    with pytest.raises(SchemaValidationError):
        validate_suggestion(bad)


def test_confidence_gate():
    s = TriageSuggestion(**VALID)
    assert confidence_gate(s, 0.7) == ("mapping_error", None)
    cat, reason = confidence_gate(s.model_copy(update={"confidence": 0.4}), 0.7)
    assert cat == NEEDS_MANUAL_REVIEW and reason.startswith("low_confidence")
