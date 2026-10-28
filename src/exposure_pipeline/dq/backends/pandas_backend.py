"""pandas implementation of the DQ backend (tests, local runs, small feeds)."""
from __future__ import annotations

import operator
from typing import Any

import pandas as pd

from exposure_pipeline.dq.backends.base import Backend, FailureSample
from exposure_pipeline.dq.rules import Rule
from exposure_pipeline.storage.db import to_python

OPS = {"<": operator.lt, "<=": operator.le, ">": operator.gt, ">=": operator.ge, "==": operator.eq, "!=": operator.ne}


def _str(s: pd.Series) -> pd.Series:
    return s.astype(str).str.strip()


class PandasBackend(Backend):
    name = "pandas"

    def count(self, df: pd.DataFrame) -> int:
        return int(len(df))

    def distinct_count(self, df: pd.DataFrame, columns: list[str]) -> int:
        return int(len(df[columns].drop_duplicates()))

    def sum(self, df: pd.DataFrame, column: str, where: dict[str, Any] | None = None) -> float:
        d = df
        for k, v in (where or {}).items():
            d = d[d[k] == v]
        return float(pd.to_numeric(d[column], errors="coerce").sum())

    def max_value(self, df: pd.DataFrame, column: str) -> Any:
        s = df[column].dropna()
        return None if s.empty else to_python(s.astype(str).max())

    def distinct_values(self, df: pd.DataFrame, column: str) -> list[Any]:
        return sorted(_str(df[column].dropna()).unique().tolist())

    def failing_rows(self, rule: Rule, df: pd.DataFrame, ref_df: pd.DataFrame | None = None, values: list[str] | None = None) -> pd.DataFrame:
        p, check = rule.params, rule.check
        if check == "not_null":
            s = df[p["column"]]
            mask = s.isna() | _str(s).eq("")
        elif check == "accepted_values":
            s = df[p["column"]]
            mask = s.notna() & ~_str(s).isin([str(v) for v in values or []])
        elif check == "pattern":
            s = df[p["column"]]
            mask = s.notna() & ~_str(s).str.fullmatch(p["regex"]).fillna(False)
        elif check == "range":
            s = df[p["column"]]
            num = pd.to_numeric(s, errors="coerce")
            bad = num.isna()  # non-null but non-numeric is a failure
            if p.get("min") is not None:
                bad |= num < float(p["min"])
            if p.get("max") is not None:
                bad |= num > float(p["max"])
            mask = s.notna() & bad
        elif check == "compare_columns":
            left = pd.to_numeric(df[p["left"]], errors="coerce")
            right = pd.to_numeric(df[p["right"]], errors="coerce")
            mask = left.notna() & right.notna() & ~OPS[p["op"]](left, right)
        elif check == "unique":
            cols = list(p["columns"])
            mask = df.duplicated(subset=cols, keep=False) & df[cols].notna().all(axis=1)
        elif check == "foreign_key":
            s = df[p["column"]]
            ref_values = set(_str(ref_df[p["ref_column"]].dropna()))
            mask = s.notna() & ~_str(s).isin(ref_values)
        else:  # pragma: no cover - guarded by rule loader
            raise ValueError(f"{check} is not a record-level check")
        return df[mask.fillna(False).astype(bool)]

    def summarize(self, failing: pd.DataFrame, key: tuple[str, ...], n_keys: int, n_records: int) -> FailureSample:
        if failing.empty:
            return FailureSample(0)
        key_cols = [k for k in key if k in failing.columns]
        keys: list[str] = []
        if key_cols:
            keys = failing[key_cols].astype(str).agg("|".join, axis=1).drop_duplicates().head(n_keys).tolist()
        sample = [{k: to_python(v) for k, v in rec.items()} for rec in failing.head(n_records).to_dict("records")]
        return FailureSample(int(len(failing)), keys, sample)
