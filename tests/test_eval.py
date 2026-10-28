"""Ground-truth derivation and metrics for the triage evaluation."""
from __future__ import annotations

import json

import pandas as pd
import pytest

from exposure_pipeline.reporting import update_readme_section
from exposure_pipeline.triage.eval import compute_metrics, derive_ground_truth, load_hand_labels, metrics_markdown

LABELS = pd.DataFrame([
    {"business_date": "2026-01-07", "feed": "facilities", "record_key": "FAC1", "level": "record", "error_type": "mapping_error"},
    {"business_date": "2026-01-07", "feed": "facilities", "record_key": "FAC2", "level": "record", "error_type": "mapping_error"},
    {"business_date": "2026-01-07", "feed": "facilities", "record_key": "FAC3", "level": "record", "error_type": "missing_source_data"},
    {"business_date": "2026-01-07", "feed": "facilities", "record_key": "*", "level": "feed", "error_type": "unit_or_fx_mismatch"},
])


def test_majority_record_label_then_feed_fallback():
    breaks = pd.DataFrame([
        {"result_id": "a", "business_date": "2026-01-07", "feed": "facilities", "failed_keys": json.dumps(["FAC1", "FAC2", "FAC3"])},
        {"result_id": "b", "business_date": "2026-01-07", "feed": "facilities", "failed_keys": "[]"},
        {"result_id": "c", "business_date": "2026-01-08", "feed": "facilities", "failed_keys": "[]"},
    ])
    truth = derive_ground_truth(breaks, LABELS)
    assert truth["a"] == "mapping_error"
    assert truth["b"] == "unit_or_fx_mismatch"
    assert truth["c"] is None


def test_metrics_count_fallbacks_as_disagreement():
    df = pd.DataFrame({
        "truth": ["mapping_error", "duplicate", "stale_feed", "duplicate"],
        "suggested_category": ["mapping_error", "duplicate", "needs_manual_review", "stale_feed"],
        "llm_category": ["mapping_error", "duplicate", "stale_feed", "stale_feed"],
        "confidence": [0.9, 0.8, 0.5, 0.75],
        "fallback_reason": [None, None, "low_confidence (0.50 < 0.70)", None],
        "latency_ms": [800.0, 900.0, 1000.0, 1100.0],
        "input_tokens": [800, 820, 790, 810], "output_tokens": [90, 95, 80, 85],
        "cost_usd": [None] * 4,
    })
    m = compute_metrics(df, "truth")
    assert m["n"] == 4
    assert m["agreement"] == pytest.approx(0.5)
    assert m["coverage"] == pytest.approx(0.75)
    assert m["agreement_on_auto"] == pytest.approx(2 / 3)
    assert m["raw_model_agreement"] == pytest.approx(0.75)
    assert m["per_class"]["duplicate"]["recall"] == pytest.approx(0.5)
    assert int(m["confusion"].loc["stale_feed", "needs_manual_review"]) == 1
    md = metrics_markdown("Test", m)
    assert "50.0%" in md and "Confusion matrix" in md


def test_metrics_with_no_data():
    assert compute_metrics(pd.DataFrame({"truth": [], "suggested_category": [], "llm_category": []}), "truth") == {"n": 0}


def test_hand_labels_template_parses_and_validates(tmp_path):
    from exposure_pipeline.triage.eval import HAND_LABELS

    assert load_hand_labels(HAND_LABELS).empty  # shipped template has only a header
    p = tmp_path / "h.csv"
    p.write_text("# comment\nbusiness_date,rule_id,analyst_category,analyst,notes\n2026-01-07,FAC_REF_001,mapping_error,ms,\n")
    s = load_hand_labels(p)
    assert s["2026-01-07:FAC_REF_001"] == "mapping_error"
    p.write_text("business_date,rule_id,analyst_category,analyst,notes\n2026-01-07,FAC_REF_001,typo,ms,\n")
    with pytest.raises(ValueError):
        load_hand_labels(p)


def test_readme_placeholder_update(tmp_path):
    readme = tmp_path / "README.md"
    readme.write_text("x\n<!-- RESULTS:demo:start -->\nold\n<!-- RESULTS:demo:end -->\ny\n")
    assert update_readme_section("demo", "| new |", readme)
    assert "| new |" in readme.read_text() and "old" not in readme.read_text()
    assert not update_readme_section("missing", "z", readme)
