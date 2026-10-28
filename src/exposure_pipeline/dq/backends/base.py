"""Backend interface: the small set of primitives the engine needs from a DataFrame library.

Design choice: rule *semantics* (what counts as a failure, thresholds, reconciliation
tolerances, result shape) live once in ``engine.py``; a backend only knows how to filter
failing rows and compute counts/sums. That keeps pandas (tests, laptop) and Spark
(production) behaviourally identical and makes adding e.g. a Polars backend cheap.
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any

from exposure_pipeline.dq.rules import Rule


@dataclass
class FailureSample:
    failed_count: int
    failed_keys: list[str] = field(default_factory=list)
    sample_records: list[dict[str, Any]] = field(default_factory=list)


class Backend(ABC):
    name: str = "abstract"

    @abstractmethod
    def count(self, df: Any) -> int: ...

    @abstractmethod
    def distinct_count(self, df: Any, columns: list[str]) -> int: ...

    @abstractmethod
    def sum(self, df: Any, column: str, where: dict[str, Any] | None = None) -> float: ...

    @abstractmethod
    def max_value(self, df: Any, column: str) -> Any: ...

    @abstractmethod
    def distinct_values(self, df: Any, column: str) -> list[Any]: ...

    @abstractmethod
    def failing_rows(self, rule: Rule, df: Any, ref_df: Any = None, values: list[str] | None = None) -> Any:
        """Return the subset of ``df`` violating a record-level rule."""

    @abstractmethod
    def summarize(self, failing: Any, key: tuple[str, ...], n_keys: int, n_records: int) -> FailureSample: ...
