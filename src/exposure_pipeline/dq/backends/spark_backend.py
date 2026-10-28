"""PySpark implementation of the DQ backend.

Every check is expressed with the DataFrame API (no Python UDFs), so evaluation runs in
the JVM on executors and only counts + a handful of sample rows come back to the driver.
Failing rows are cached once per rule because they are counted and sampled.
"""
from __future__ import annotations

import operator
from typing import Any

from pyspark.sql import DataFrame, Window
from pyspark.sql import functions as F

from exposure_pipeline.dq.backends.base import Backend, FailureSample
from exposure_pipeline.dq.rules import Rule
from exposure_pipeline.storage.db import to_python

OPS = {"<": operator.lt, "<=": operator.le, ">": operator.gt, ">=": operator.ge, "==": operator.eq, "!=": operator.ne}


def _trimmed(column: str):
    return F.trim(F.col(column).cast("string"))


class SparkBackend(Backend):
    name = "spark"

    def count(self, df: DataFrame) -> int:
        return int(df.count())

    def distinct_count(self, df: DataFrame, columns: list[str]) -> int:
        return int(df.select(*columns).distinct().count())

    def sum(self, df: DataFrame, column: str, where: dict[str, Any] | None = None) -> float:
        d = df
        for k, v in (where or {}).items():
            d = d.filter(F.col(k) == F.lit(v))
        value = d.agg(F.sum(F.col(column).cast("double"))).first()[0]
        return float(value or 0.0)

    def max_value(self, df: DataFrame, column: str) -> Any:
        value = df.agg(F.max(F.col(column).cast("string"))).first()[0]
        return to_python(value)

    def distinct_values(self, df: DataFrame, column: str) -> list[Any]:
        rows = df.select(_trimmed(column).alias("v")).where(F.col("v").isNotNull()).distinct().collect()
        return sorted(r["v"] for r in rows)

    def failing_rows(self, rule: Rule, df: DataFrame, ref_df: DataFrame | None = None, values: list[str] | None = None) -> DataFrame:
        p, check = rule.params, rule.check
        if check == "not_null":
            c = F.col(p["column"])
            return df.filter(c.isNull() | (_trimmed(p["column"]) == ""))
        if check == "accepted_values":
            c = F.col(p["column"])
            return df.filter(c.isNotNull() & ~_trimmed(p["column"]).isin([str(v) for v in values or []]))
        if check == "pattern":
            c = F.col(p["column"])
            return df.filter(c.isNotNull() & ~_trimmed(p["column"]).rlike(f"^(?:{p['regex']})$"))
        if check == "range":
            c = F.col(p["column"])
            num = c.cast("double")
            bad = num.isNull()
            if p.get("min") is not None:
                bad = bad | (num < F.lit(float(p["min"])))
            if p.get("max") is not None:
                bad = bad | (num > F.lit(float(p["max"])))
            return df.filter(c.isNotNull() & bad)
        if check == "compare_columns":
            left, right = F.col(p["left"]).cast("double"), F.col(p["right"]).cast("double")
            return df.filter(left.isNotNull() & right.isNotNull() & ~OPS[p["op"]](left, right))
        if check == "unique":
            cols = list(p["columns"])
            w = Window.partitionBy(*cols)
            not_null = F.lit(True)
            for c in cols:
                not_null = not_null & F.col(c).isNotNull()
            return (df.withColumn("_dq_n", F.count(F.lit(1)).over(w))
                    .filter((F.col("_dq_n") > 1) & not_null).drop("_dq_n"))
        if check == "foreign_key":
            ref = ref_df.select(_trimmed(p["ref_column"]).alias("_dq_ref")).where(F.col("_dq_ref").isNotNull()).distinct()
            return (df.filter(F.col(p["column"]).isNotNull())
                    .join(F.broadcast(ref) if rule.params.get("broadcast", True) else ref,
                          _trimmed(p["column"]) == F.col("_dq_ref"), "left_anti"))
        raise ValueError(f"{check} is not a record-level check")

    def summarize(self, failing: DataFrame, key: tuple[str, ...], n_keys: int, n_records: int) -> FailureSample:
        failing = failing.cache()
        try:
            n = int(failing.count())
            if n == 0:
                return FailureSample(0)
            key_cols = [k for k in key if k in failing.columns]
            keys: list[str] = []
            if key_cols:
                key_expr = F.concat_ws("|", *[F.col(k).cast("string") for k in key_cols]).alias("_k")
                keys = [r["_k"] for r in failing.select(key_expr).distinct().limit(n_keys).collect()]
            sample = [{k: to_python(v) for k, v in r.asDict().items()} for r in failing.limit(n_records).collect()]
            return FailureSample(n, keys, sample)
        finally:
            failing.unpersist()
