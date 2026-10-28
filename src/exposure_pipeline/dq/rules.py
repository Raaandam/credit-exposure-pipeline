"""Rule model and YAML loader.

Rules are data, not code: adding a completeness check for a new column is a YAML change
reviewed like any config, with no engine release. The loader validates the catalogue up
front (unknown check types, missing parameters, duplicate ids) so a bad rule file fails the
deployment, not the 6am production run.

YAML shape::

    defaults: {severity: high, threshold: 0.0}
    rules:
      - id: FAC_REF_001
        name: Facility counterparty exists in counterparty master
        dataset: raw.facilities          # which dataset the rule runs on
        feed: facilities                 # optional, derived from raw.<feed>
        dimension: referential_integrity
        check: foreign_key
        column: counterparty_id
        ref_dataset: raw.counterparties
        ref_column: counterparty_id
        description: ...

Every key not in the common set is passed to the check as a parameter.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

from exposure_pipeline import reference as ref

RULES_DIR = Path(__file__).parent / "rules"

# check type -> (level, required params)
CHECKS: dict[str, tuple[str, tuple[str, ...]]] = {
    "not_null": ("record", ("column",)),
    "accepted_values": ("record", ("column",)),
    "pattern": ("record", ("column", "regex")),
    "range": ("record", ("column",)),
    "compare_columns": ("record", ("left", "op", "right")),
    "unique": ("record", ("columns",)),
    "foreign_key": ("record", ("column", "ref_dataset", "ref_column")),
    "min_row_count": ("aggregate", ("min_rows",)),
    "freshness": ("aggregate", ("column", "max_lag_days")),
    "count_reconciliation": ("aggregate", ("source", "target")),
    "sum_reconciliation": ("aggregate", ("source", "target")),
}
DIMENSIONS = {"completeness", "validity", "uniqueness", "referential_integrity", "timeliness", "reconciliation"}
SEVERITIES = {"low", "medium", "high", "critical"}
COMMON_KEYS = {"id", "name", "dataset", "feed", "dimension", "check", "severity", "threshold", "description", "key", "enabled"}


class RuleConfigError(ValueError):
    pass


@dataclass(frozen=True)
class Rule:
    rule_id: str
    name: str
    dataset: str
    feed: str
    dimension: str
    check: str
    severity: str = "high"
    threshold: float = 0.0  # tolerated failure rate for record-level checks
    description: str = ""
    key: tuple[str, ...] = ()
    params: dict[str, Any] = field(default_factory=dict)

    @property
    def level(self) -> str:
        return CHECKS[self.check][0]

    def as_prompt_dict(self) -> dict[str, Any]:
        """Compact, LLM-friendly description of the rule (used by triage context)."""
        return {"rule_id": self.rule_id, "name": self.name, "dimension": self.dimension, "check": self.check,
                "dataset": self.dataset, "description": self.description.strip(), "parameters": self.params}


def _parse(raw: dict, defaults: dict, source: str) -> Rule:
    merged = {**defaults, **raw}
    rid = merged.get("id")
    for required in ("id", "name", "dataset", "dimension", "check"):
        if not merged.get(required):
            raise RuleConfigError(f"{source}: rule {rid!r} missing '{required}'")
    check = merged["check"]
    if check not in CHECKS:
        raise RuleConfigError(f"{source}: rule {rid} has unknown check '{check}'")
    if merged["dimension"] not in DIMENSIONS:
        raise RuleConfigError(f"{source}: rule {rid} has unknown dimension '{merged['dimension']}'")
    severity = merged.get("severity", "high")
    if severity not in SEVERITIES:
        raise RuleConfigError(f"{source}: rule {rid} has unknown severity '{severity}'")
    params = {k: v for k, v in merged.items() if k not in COMMON_KEYS}
    missing = [p for p in CHECKS[check][1] if p not in params]
    if missing:
        raise RuleConfigError(f"{source}: rule {rid} ({check}) missing params {missing}")
    if check == "accepted_values" and "values" not in params and "values_from" not in params:
        raise RuleConfigError(f"{source}: rule {rid} needs 'values' or 'values_from'")

    dataset = merged["dataset"]
    feed = merged.get("feed") or (dataset.split(".", 1)[1] if dataset.startswith("raw.") else None)
    if not feed:
        raise RuleConfigError(f"{source}: rule {rid} on {dataset} needs an explicit 'feed'")
    key = merged.get("key") or ref.FEED_KEYS.get(feed, [])
    return Rule(
        rule_id=rid, name=merged["name"], dataset=dataset, feed=feed, dimension=merged["dimension"], check=check,
        severity=severity, threshold=float(merged.get("threshold", 0.0)), description=merged.get("description", ""),
        key=tuple(key), params=params,
    )


def load_rules(path: Path | str = RULES_DIR, include_disabled: bool = False) -> list[Rule]:
    path = Path(path)
    files = sorted(path.glob("*.yaml")) if path.is_dir() else [path]
    rules: list[Rule] = []
    seen: set[str] = set()
    for f in files:
        doc = yaml.safe_load(f.read_text(encoding="utf-8")) or {}
        defaults = doc.get("defaults", {})
        for raw in doc.get("rules", []):
            if not include_disabled and raw.get("enabled", True) is False:
                continue
            rule = _parse(raw, defaults, f.name)
            if rule.rule_id in seen:
                raise RuleConfigError(f"duplicate rule id {rule.rule_id} in {f.name}")
            seen.add(rule.rule_id)
            rules.append(rule)
    return rules
