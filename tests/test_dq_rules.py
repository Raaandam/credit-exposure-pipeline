"""DQ engine + rule catalogue on small pandas fixtures."""
from __future__ import annotations

from collections import Counter

import pytest
import yaml

from exposure_pipeline.dq.backends.pandas_backend import PandasBackend
from exposure_pipeline.dq.engine import DQEngine
from exposure_pipeline.dq.rules import CHECKS, RULES_DIR, Rule, RuleConfigError, load_rules
from exposure_pipeline.transform.pandas_impl import curate_all


def _rule(check: str, dataset: str = "raw.facilities", **params) -> Rule:
    feed = dataset.split(".", 1)[1] if dataset.startswith("raw.") else "facilities"
    from exposure_pipeline import reference as ref

    return Rule(rule_id=f"T_{check}", name=check, dataset=dataset, feed=feed, dimension="validity", check=check,
                key=tuple(ref.FEED_KEYS.get(feed, [])), params=params)


def _run(rule: Rule, datasets: dict, bd):
    return DQEngine([rule], PandasBackend()).run(datasets, bd)[0]


# ----------------------------------------------------------------------------- catalogue
def test_catalogue_has_25_plus_rules_across_all_dimensions():
    rules = load_rules()
    assert len(rules) >= 25
    dims = Counter(r.dimension for r in rules)
    for d in ("completeness", "validity", "uniqueness", "referential_integrity", "reconciliation"):
        assert dims[d] >= 1, d
    assert len({r.rule_id for r in rules}) == len(rules)
    assert all(r.check in CHECKS for r in rules)


def test_loader_rejects_bad_rules(tmp_path):
    bad = {"rules": [{"id": "X1", "name": "x", "dataset": "raw.facilities", "dimension": "validity", "check": "not_null"}]}
    p = tmp_path / "bad.yaml"
    p.write_text(yaml.safe_dump(bad))
    with pytest.raises(RuleConfigError, match="missing params"):
        load_rules(p)
    bad["rules"][0].update(check="no_such_check", column="x")
    p.write_text(yaml.safe_dump(bad))
    with pytest.raises(RuleConfigError, match="unknown check"):
        load_rules(p)


def test_loader_rejects_duplicate_ids(tmp_path):
    r = {"id": "X1", "name": "x", "dataset": "raw.facilities", "dimension": "validity", "check": "not_null", "column": "c"}
    p = tmp_path / "dup.yaml"
    p.write_text(yaml.safe_dump({"rules": [r, r]}))
    with pytest.raises(RuleConfigError, match="duplicate"):
        load_rules(p)


def test_curated_rules_need_explicit_feed(tmp_path):
    r = {"id": "X1", "name": "x", "dataset": "curated.rating_change", "dimension": "validity", "check": "range", "column": "c"}
    p = tmp_path / "r.yaml"
    p.write_text(yaml.safe_dump({"rules": [r]}))
    with pytest.raises(RuleConfigError, match="explicit 'feed'"):
        load_rules(p)


# ----------------------------------------------------------------------------- record checks
def test_not_null(raw_datasets, bd):
    res = _run(_rule("not_null", "raw.counterparties", column="internal_rating"), raw_datasets, bd)
    assert res.status == "FAIL" and res.failed_count == 1 and res.failed_keys == ["CP000003"]
    assert res.sample_records[0]["counterparty_id"] == "CP000003"


def test_not_null_threshold_tolerates_low_failure_rate(raw_datasets, bd):
    rule = _rule("not_null", "raw.counterparties", column="internal_rating")
    rule = Rule(**{**rule.__dict__, "threshold": 0.5})
    assert _run(rule, raw_datasets, bd).status == "PASS"


def test_accepted_values_ignores_nulls(raw_datasets, bd):
    res = _run(_rule("accepted_values", "raw.counterparties", column="internal_rating",
                     values=["AAA", "AA", "A", "BBB", "BB", "B", "CCC", "D"]), raw_datasets, bd)
    assert res.failed_count == 2  # the two Baa2 rows; the NULL belongs to completeness
    assert res.failed_keys == ["CP000004"]


def test_accepted_values_from_reference_dataset(raw_datasets, bd):
    res = _run(_rule("accepted_values", column="currency", values_from={"dataset": "raw.fx_rates", "column": "currency"}),
               raw_datasets, bd)
    assert res.failed_keys == ["FAC0000003"]  # GBX


def test_pattern(raw_datasets, bd):
    res = _run(_rule("pattern", "raw.counterparties", column="country", regex="[A-Z]{2}"), raw_datasets, bd)
    assert res.failed_keys == ["CP000003"]


def test_range_catches_out_of_bounds_and_non_numeric(raw_datasets, bd):
    ds = dict(raw_datasets)
    fac = ds["raw.facilities"].copy()
    fac.loc[0, "limit_amount"] = "N/A"
    ds["raw.facilities"] = fac
    res = _run(_rule("range", column="limit_amount", min=10000, max=2e9), ds, bd)
    assert set(res.failed_keys) == {"FAC0000001", "FAC0000003"}


def test_compare_columns(raw_datasets, bd):
    res = _run(_rule("compare_columns", left="drawn_amount", op="<=", right="limit_amount"), raw_datasets, bd)
    assert res.failed_keys == ["FAC0000004"]


def test_unique_reports_all_copies(raw_datasets, bd):
    res = _run(_rule("unique", columns=["facility_id"]), raw_datasets, bd)
    assert res.failed_count == 2 and res.failed_keys == ["FAC0000005"]


def test_foreign_key(raw_datasets, bd):
    res = _run(_rule("foreign_key", column="counterparty_id", ref_dataset="raw.counterparties", ref_column="counterparty_id"),
               raw_datasets, bd)
    assert res.failed_keys == ["FAC0000004"]


# ----------------------------------------------------------------------------- aggregate checks
def test_freshness_flags_stale_feed(raw_datasets, bd):
    ds = dict(raw_datasets)
    ds["raw.collateral"] = ds["raw.collateral"].assign(as_of_date="2026-01-06")
    res = _run(_rule("freshness", "raw.collateral", column="as_of_date", max_lag_days=0), ds, bd)
    assert res.status == "FAIL" and "lag 1" in res.message
    ok = _run(_rule("freshness", "raw.facilities", column="as_of_date", max_lag_days=0), raw_datasets, bd)
    assert ok.status == "PASS"


def test_count_reconciliation_against_manifest(raw_datasets, manifest, bd):
    ds = {**raw_datasets, "manifest": manifest}
    res = _run(_rule("count_reconciliation", source={"manifest": "facilities", "field": "record_count"},
                     target={"dataset": "raw.facilities"}), ds, bd)
    assert res.status == "FAIL" and res.failed_count == 1


def test_sum_reconciliation_with_tolerance(raw_datasets, manifest, bd):
    ds = {**raw_datasets, "manifest": manifest}
    drawn = _run(_rule("sum_reconciliation", source={"manifest": "facilities", "field": "control_total_drawn"},
                       target={"dataset": "raw.facilities", "column": "drawn_amount"}, tolerance_pct=1e-6), ds, bd)
    limit = _run(_rule("sum_reconciliation", source={"manifest": "facilities", "field": "control_total_limit"},
                       target={"dataset": "raw.facilities", "column": "limit_amount"}, tolerance_pct=1e-9), ds, bd)
    assert drawn.status == "PASS" and limit.status == "FAIL"


def test_missing_dataset_is_error_not_crash(raw_datasets, bd):
    ds = {k: v for k, v in raw_datasets.items() if k != "raw.collateral"}
    res = _run(_rule("not_null", "raw.collateral", column="market_value"), ds, bd)
    assert res.status == "ERROR" and "raw.collateral" in res.message


def test_missing_manifest_field_is_error(raw_datasets, bd):
    res = _run(_rule("count_reconciliation", source={"manifest": "facilities", "field": "record_count"},
                     target={"dataset": "raw.facilities"}), {**raw_datasets, "manifest": {}}, bd)
    assert res.status == "ERROR"


# ----------------------------------------------------------------------------- full catalogue run
def test_full_catalogue_on_fixture(raw_datasets, manifest, bd):
    datasets = {**raw_datasets, **curate_all(raw_datasets, bd), "manifest": manifest}
    results = {r.rule_id: r for r in DQEngine(load_rules(), PandasBackend()).run(datasets, bd)}
    assert not [r for r in results.values() if r.status == "ERROR"], "every rule should evaluate on a complete fixture"
    expected_fail = {"CP_COMP_001", "CP_VAL_001", "CP_VAL_002", "CP_VAL_003", "CP_UNQ_001", "FAC_REF_001", "FAC_UNQ_001",
                     "FAC_VAL_001", "FAC_VAL_004", "FAC_VAL_005", "FAC_COMP_001", "COL_REF_001", "COL_VAL_001",
                     "COL_COMP_001", "RAT_VAL_002", "REC_FAC_CNT_001", "REC_FAC_SUM_002"}
    failed = {k for k, r in results.items() if r.status == "FAIL"}
    assert expected_fail <= failed, expected_fail - failed
    # Source-to-target checks must pass: curation only removes duplicates, and every
    # mart dimension sums to the curated total.
    for rid in ("S2T_CP_CNT_001", "S2T_FAC_CNT_001", "S2T_EXP_SUM_001", "S2T_EXP_SUM_002", "S2T_EXP_SUM_003", "S2T_EXP_SUM_004"):
        assert results[rid].status == "PASS", (rid, results[rid].observed_value)


def test_rule_files_present():
    assert sorted(p.stem for p in RULES_DIR.glob("*.yaml")) == [
        "completeness", "reconciliation", "referential_integrity", "timeliness", "uniqueness", "validity"]
