"""Config-driven data-quality engine.

Usage::

    rules = load_rules()                                  # dq/rules/*.yaml
    engine = DQEngine(rules, PandasBackend())             # or SparkBackend()
    results = engine.run(datasets, business_date)         # list[DQResult]

``datasets`` maps logical names ("raw.facilities", "curated.facility_exposure",
"mart.exposure_aggregate", ...) to DataFrames of the backend's type, plus "manifest":
the source control file as ``{feed: {...record_count, control totals...}}``.

Semantics:
* Record-level checks FAIL when ``failed_count > 0`` and ``failure_rate > threshold``
  (threshold defaults to 0 = zero tolerance; set e.g. 0.001 for noisy optional fields).
* Aggregate checks (row counts, freshness, reconciliations) FAIL on any breach beyond the
  configured tolerance.
* A rule never crashes the run: a missing dataset or an evaluation error is recorded as
  status ERROR (a missing feed is itself a break someone must look at).
"""
from __future__ import annotations

import logging
import uuid
from dataclasses import asdict, dataclass, field
from datetime import date, datetime, timedelta, timezone
from typing import Any, Mapping

import pandas as pd

from exposure_pipeline.dq.backends.base import Backend, FailureSample
from exposure_pipeline.dq.rules import Rule

log = logging.getLogger(__name__)


class DatasetMissing(KeyError):
    pass


@dataclass
class DQResult:
    result_id: str
    run_id: str
    business_date: date
    rule_id: str
    rule_name: str
    feed: str
    dataset: str
    dimension: str
    check_type: str
    severity: str
    status: str
    total_count: int | None = None
    failed_count: int | None = None
    failure_rate: float | None = None
    observed_value: str | None = None
    expected_value: str | None = None
    message: str = ""
    failed_keys: list[str] = field(default_factory=list)
    sample_records: list[dict[str, Any]] = field(default_factory=list)
    executed_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc).replace(tzinfo=None))

    @property
    def is_break(self) -> bool:
        return self.status in {"FAIL", "ERROR"}

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _fmt(x: float | int | None) -> str | None:
    if x is None:
        return None
    return f"{x:,.2f}" if isinstance(x, float) else f"{x:,}"


class DQEngine:
    def __init__(self, rules: list[Rule], backend: Backend, sample_keys: int = 20, sample_records: int = 5):
        self.rules = rules
        self.backend = backend
        self.sample_keys = sample_keys
        self.sample_records = sample_records

    # ------------------------------------------------------------------ public
    def run(self, datasets: Mapping[str, Any], business_date: date, run_id: str | None = None) -> list[DQResult]:
        run_id = run_id or uuid.uuid4().hex
        results = [self._run_rule(rule, datasets, business_date, run_id) for rule in self.rules]
        n_breaks = sum(r.is_break for r in results)
        log.info("DQ run %s for %s: %d rules, %d breaks (%s backend)", run_id, business_date, len(results), n_breaks, self.backend.name)
        return results

    # ------------------------------------------------------------------ internals
    def _run_rule(self, rule: Rule, datasets: Mapping[str, Any], bd: date, run_id: str) -> DQResult:
        res = DQResult(
            result_id=f"{bd.isoformat()}:{rule.rule_id}", run_id=run_id, business_date=bd, rule_id=rule.rule_id,
            rule_name=rule.name, feed=rule.feed, dataset=rule.dataset, dimension=rule.dimension, check_type=rule.check,
            severity=rule.severity, status="PASS",
        )
        try:
            if rule.level == "record":
                self._record_check(rule, datasets, res)
            else:
                self._aggregate_check(rule, datasets, bd, res)
        except DatasetMissing as exc:
            res.status, res.message = "ERROR", f"dataset not available: {exc.args[0]}"
        except Exception as exc:  # noqa: BLE001 - one bad rule must not stop the run
            log.exception("rule %s failed to evaluate", rule.rule_id)
            res.status, res.message = "ERROR", f"{type(exc).__name__}: {exc}"
        return res

    @staticmethod
    def _dataset(datasets: Mapping[str, Any], name: str) -> Any:
        if name not in datasets or datasets[name] is None:
            raise DatasetMissing(name)
        return datasets[name]

    def _record_check(self, rule: Rule, datasets: Mapping[str, Any], res: DQResult) -> None:
        df = self._dataset(datasets, rule.dataset)
        p = rule.params
        ref_df = self._dataset(datasets, p["ref_dataset"]) if rule.check == "foreign_key" else None
        values = None
        if rule.check == "accepted_values":
            if "values_from" in p:
                src = p["values_from"]
                values = self.backend.distinct_values(self._dataset(datasets, src["dataset"]), src["column"])
            else:
                values = [str(v) for v in p["values"]]
        total = self.backend.count(df)
        sample: FailureSample = self.backend.summarize(
            self.backend.failing_rows(rule, df, ref_df=ref_df, values=values), rule.key, self.sample_keys, self.sample_records)
        res.total_count, res.failed_count = total, sample.failed_count
        res.failure_rate = (sample.failed_count / total) if total else 0.0
        res.failed_keys, res.sample_records = sample.failed_keys, sample.sample_records
        res.expected_value = f"failure rate <= {rule.threshold:.4%}"
        res.observed_value = f"{sample.failed_count} of {total} rows ({res.failure_rate:.4%})"
        if sample.failed_count > 0 and res.failure_rate > rule.threshold:
            res.status = "FAIL"
            res.message = f"{sample.failed_count} rows violate {rule.check}"

    def _measure(self, spec: dict, datasets: Mapping[str, Any], kind: str) -> float | int:
        """Evaluate one side of a reconciliation: a manifest field or a dataset metric."""
        if "manifest" in spec:
            manifest = self._dataset(datasets, "manifest")
            feed = manifest.get(spec["manifest"]) if isinstance(manifest, Mapping) else None
            if not feed or spec["field"] not in feed:
                raise DatasetMissing(f"manifest.{spec['manifest']}.{spec['field']}")
            return feed[spec["field"]]
        df = self._dataset(datasets, spec["dataset"])
        if kind == "sum":
            return self.backend.sum(df, spec["column"], spec.get("where"))
        if spec.get("distinct"):
            return self.backend.distinct_count(df, list(spec["distinct"]))
        return self.backend.count(df)

    def _aggregate_check(self, rule: Rule, datasets: Mapping[str, Any], bd: date, res: DQResult) -> None:
        p = rule.params
        if rule.check == "min_row_count":
            n = self.backend.count(self._dataset(datasets, rule.dataset))
            res.total_count, res.failed_count = n, int(n < int(p["min_rows"]))
            res.observed_value, res.expected_value = f"{n:,} rows", f">= {int(p['min_rows']):,} rows"
            if res.failed_count:
                res.status, res.message = "FAIL", "feed empty or below minimum volume"
            return

        if rule.check == "freshness":
            df = self._dataset(datasets, rule.dataset)
            latest = self.backend.max_value(df, p["column"])
            earliest_ok = bd - timedelta(days=int(p["max_lag_days"]))
            latest_date = pd.to_datetime(latest).date() if latest is not None else None
            res.total_count = self.backend.count(df)
            res.observed_value = f"max({p['column']}) = {latest_date}"
            res.expected_value = f">= {earliest_ok.isoformat()}"
            stale = latest_date is None or latest_date < earliest_ok
            res.failed_count = int(stale)
            if stale:
                lag = (bd - latest_date).days if latest_date else None
                res.status, res.message = "FAIL", f"feed is stale (lag {lag} days)"
            return

        kind = "sum" if rule.check == "sum_reconciliation" else "count"
        source = self._measure(p["source"], datasets, kind)
        target = self._measure(p["target"], datasets, kind)
        diff = float(target) - float(source)
        res.observed_value = f"target={_fmt(target)} source={_fmt(source)} diff={_fmt(diff)}"
        if kind == "count":
            tol = float(p.get("tolerance_abs", 0))
            res.total_count, res.failed_count = int(source), int(abs(diff))
            res.expected_value = f"|diff| <= {tol:g} rows"
            breach = abs(diff) > tol
        else:
            tol_pct, tol_abs = float(p.get("tolerance_pct", 0.0)), float(p.get("tolerance_abs", 0.01))
            allowed = max(tol_abs, abs(float(source)) * tol_pct)
            res.total_count, res.failed_count = 1, int(abs(diff) > allowed)
            res.expected_value = f"|diff| <= {allowed:,.2f} ({tol_pct:.4%} of source)"
            breach = abs(diff) > allowed
        res.failure_rate = (abs(diff) / abs(float(source))) if float(source) else (1.0 if diff else 0.0)
        if breach:
            res.status, res.message = "FAIL", f"{rule.check} breach: target differs from source by {_fmt(diff)}"
