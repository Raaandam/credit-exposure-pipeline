"""Spark backend/transforms must agree with the pandas reference (skipped without pyspark/Java)."""
from __future__ import annotations

import pytest

pyspark = pytest.importorskip("pyspark")

from exposure_pipeline.dq.backends.pandas_backend import PandasBackend  # noqa: E402
from exposure_pipeline.dq.engine import DQEngine  # noqa: E402
from exposure_pipeline.dq.rules import load_rules  # noqa: E402
from exposure_pipeline.transform import pandas_impl  # noqa: E402


@pytest.fixture(scope="module")
def spark():
    from pyspark.sql import SparkSession

    try:
        s = SparkSession.builder.master("local[1]").appName("tests").config("spark.sql.shuffle.partitions", "2").getOrCreate()
    except Exception as exc:  # noqa: BLE001 - e.g. no Java runtime
        pytest.skip(f"Spark unavailable: {exc}")
    yield s
    s.stop()


def _to_spark(spark, df):
    return spark.createDataFrame(df.astype(object).where(df.notna(), None).astype(str).replace("None", None))


def test_dq_results_match_pandas(spark, raw_datasets, manifest, bd):
    from exposure_pipeline.dq.backends.spark_backend import SparkBackend
    from exposure_pipeline.spark_jobs import transforms

    rules = load_rules()
    pd_cur = pandas_impl.curate_all(raw_datasets, bd)
    pd_res = {r.rule_id: (r.status, r.failed_count) for r in DQEngine(rules, PandasBackend()).run(
        {**raw_datasets, **pd_cur, "manifest": manifest}, bd)}

    sp_raw = {k: _to_spark(spark, v) for k, v in raw_datasets.items()}
    sp_cur = transforms.curate_all(sp_raw, bd)
    sp_res = {r.rule_id: (r.status, r.failed_count) for r in DQEngine(rules, SparkBackend()).run(
        {**sp_raw, **sp_cur, "manifest": manifest}, bd)}
    assert sp_res == pd_res


def test_aggregates_match_pandas(spark, raw_datasets, bd):
    from exposure_pipeline.spark_jobs import transforms

    pd_agg = pandas_impl.curate_all(raw_datasets, bd)["mart.exposure_aggregate"]
    sp_agg = transforms.curate_all({k: _to_spark(spark, v) for k, v in raw_datasets.items()}, bd)["mart.exposure_aggregate"].toPandas()
    key = ["dimension", "dimension_value"]
    a = pd_agg.sort_values(key).reset_index(drop=True)
    b = sp_agg.sort_values(key).reset_index(drop=True)
    assert list(a["dimension_value"]) == list(b["dimension_value"])
    assert a["ead_chf"].sum() == pytest.approx(b["ead_chf"].sum())
