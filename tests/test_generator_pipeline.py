"""End-to-end on a small generated universe: generator -> local pipeline (SQLite) -> labels."""
from __future__ import annotations

from datetime import date

import pandas as pd
import pytest

from exposure_pipeline.dq.backends.pandas_backend import PandasBackend
from exposure_pipeline.dq.engine import DQEngine
from exposure_pipeline.dq.rules import load_rules
from exposure_pipeline.generator.generate_feeds import GeneratorConfig, business_days, generate
from exposure_pipeline.local_pipeline import run_business_date
from exposure_pipeline.storage.db import ensure_schema, get_engine, read_sql
from exposure_pipeline.triage.eval import derive_ground_truth

SCHEDULE = [None, "facilities_unit_thousands", "rating_changes_stale", "facilities_truncation"]


@pytest.fixture(scope="module")
def pipeline(tmp_path_factory):
    out = tmp_path_factory.mktemp("gen")
    generate(GeneratorConfig(out_dir=out, n_counterparties=300, n_days=4, seed=7, incident_schedule=SCHEDULE,
                             error_type_probability=1.0))
    engine = get_engine(f"sqlite:///{(out / 'test.db').as_posix()}")
    ensure_schema(engine)
    dq = DQEngine(load_rules(), PandasBackend())
    days = business_days(date(2026, 1, 5), 4)
    for d in days:
        run_business_date(d, engine, out / "raw", dq)
    labels = pd.read_csv(out / "labels" / "error_labels.csv", dtype=str, keep_default_na=False)
    return engine, labels, days, out


def test_business_days_skip_weekends():
    days = business_days(date(2026, 1, 9), 3)  # Friday
    assert [d.isoformat() for d in days] == ["2026-01-09", "2026-01-12", "2026-01-13"]


def test_clean_first_day_has_no_breaks(pipeline):
    engine, _, days, _ = pipeline
    res = read_sql(engine, "SELECT status FROM dq_results WHERE business_date = :d", d=days[0].isoformat())
    assert len(res) == len(load_rules()) and set(res["status"]) == {"PASS"}


def test_incidents_trigger_expected_rules(pipeline):
    engine, _, days, _ = pipeline
    fails = read_sql(engine, "SELECT business_date, rule_id FROM dq_results WHERE status = 'FAIL'")
    by_day = fails.groupby("business_date")["rule_id"].apply(set).to_dict()
    assert {"REC_FAC_SUM_001", "REC_FAC_SUM_002", "FAC_VAL_005"} <= by_day[days[1].isoformat()]  # unit change
    assert "FRESH_RAT_001" in by_day[days[2].isoformat()]                                        # stale rating feed
    assert {"REC_FAC_CNT_001", "COL_REF_001"} <= by_day[days[3].isoformat()]                     # truncation


def test_source_to_target_always_reconciles(pipeline):
    engine, *_ = pipeline
    s2t = read_sql(engine, "SELECT status FROM dq_results WHERE rule_id LIKE 'S2T_%'")
    assert set(s2t["status"]) == {"PASS"}


def test_rerun_is_idempotent(pipeline):
    engine, _, days, out = pipeline
    q = "SELECT COUNT(*) AS n, SUM(ead_chf) AS ead FROM cur_facility_exposure WHERE business_date = :d"
    before = read_sql(engine, q, d=days[1].isoformat())
    run_business_date(days[1], engine, out / "raw", DQEngine(load_rules(), PandasBackend()))
    after = read_sql(engine, q, d=days[1].isoformat())
    pd.testing.assert_frame_equal(before, after)
    assert read_sql(engine, "SELECT COUNT(*) AS n FROM dq_results WHERE business_date = :d", d=days[1].isoformat())["n"][0] == len(load_rules())


def test_every_break_maps_to_a_label(pipeline):
    engine, labels, *_ = pipeline
    breaks = read_sql(engine, "SELECT result_id, business_date, rule_id, feed, failed_keys FROM dq_results WHERE status = 'FAIL'")
    truth = derive_ground_truth(breaks, labels)
    coverage = truth.notna().mean()
    assert coverage > 0.9, truth[truth.isna()]
    assert set(truth.dropna()) <= {"missing_source_data", "mapping_error", "stale_feed", "duplicate",
                                   "unit_or_fx_mismatch", "genuine_business_change"}
    stale = breaks.loc[breaks["rule_id"] == "FRESH_RAT_001", "result_id"]
    assert (truth[stale] == "stale_feed").all()
