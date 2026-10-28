"""Ground-truth label log for injected data errors.

Two label levels:
* ``record`` - a specific business key in a feed was corrupted (e.g. facility FAC0001234 was
  given a non-existent counterparty id). Record-level DQ breaks are matched to these via
  their failing-key sample.
* ``feed`` - a whole-file incident (stale copy, truncated transfer, unit change in one
  source system). Aggregate checks (freshness, count/total reconciliation) have no failing
  keys, so they are matched to these instead.

The generator guarantees at most one record-level injection per (date, feed, key) and at
most one feed-level incident per (date, feed), so the mapping from a break to its ground
truth is unambiguous enough to evaluate against.
"""
from __future__ import annotations

from datetime import date
from typing import Iterable

import pandas as pd

LABEL_COLUMNS = [
    "business_date",
    "feed",
    "record_key",
    "level",
    "error_type",
    "injection",
    "column",
    "description",
]


class LabelLog:
    def __init__(self) -> None:
        self._rows: list[dict] = []

    def record(
        self,
        business_date: date,
        feed: str,
        keys: Iterable,
        error_type: str,
        injection: str,
        column: str,
        description: str,
    ) -> None:
        for key in keys:
            self._rows.append(
                {
                    "business_date": business_date.isoformat(),
                    "feed": feed,
                    "record_key": str(key),
                    "level": "record",
                    "error_type": error_type,
                    "injection": injection,
                    "column": column,
                    "description": description,
                }
            )

    def feed(self, business_date: date, feed: str, error_type: str, injection: str, description: str) -> None:
        self._rows.append(
            {
                "business_date": business_date.isoformat(),
                "feed": feed,
                "record_key": "*",
                "level": "feed",
                "error_type": error_type,
                "injection": injection,
                "column": "",
                "description": description,
            }
        )

    def keys(self, business_date: date, feed: str) -> set[str]:
        bd = business_date.isoformat()
        return {r["record_key"] for r in self._rows if r["business_date"] == bd and r["feed"] == feed}

    def drop(self, business_date: date, feed: str, level: str = "record") -> None:
        """Remove labels for a feed/day, e.g. when a stale copy replaces today's file."""
        bd = business_date.isoformat()
        self._rows = [
            r for r in self._rows if not (r["business_date"] == bd and r["feed"] == feed and r["level"] == level)
        ]

    def to_frame(self) -> pd.DataFrame:
        return pd.DataFrame(self._rows, columns=LABEL_COLUMNS)
