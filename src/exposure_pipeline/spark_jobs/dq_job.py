"""Daily DQ job on Spark: raw feeds + curated parquet -> dq_results.

Runs as its own Airflow task (trigger_rule=all_done) so that DQ results are produced even
when the exposure job failed - a missing feed shows up as ERROR/FAIL rows on the dashboard
instead of as a silent gap. Uses the same YAML rules and engine as the pandas path.
"""
from __future__ import annotations

import argparse
from datetime import date

from pyspark.sql import functions as F

from exposure_pipeline.config import get_settings
from exposure_pipeline.dq.backends.spark_backend import SparkBackend
from exposure_pipeline.dq.engine import DQEngine
from exposure_pipeline.dq.results import write_results
from exposure_pipeline.dq.rules import load_rules
from exposure_pipeline.spark_jobs.session import build_spark, read_manifest, read_raw
from exposure_pipeline.storage.blob import curated_uri
from exposure_pipeline.storage.db import ensure_schema, get_engine
from exposure_pipeline.storage.tables import TABLES_BY_DATASET


def run(business_date: date) -> dict:
    settings = get_settings()
    spark = build_spark(f"credit-dq-{business_date}", settings)
    datasets: dict = dict(read_raw(spark, settings, business_date))
    for name, table in TABLES_BY_DATASET.items():
        try:
            df = spark.read.parquet(curated_uri(settings, table.name)).filter(F.col("business_date") == F.lit(business_date.isoformat()).cast("date"))
            datasets[name] = df.cache()
        except Exception as exc:  # noqa: BLE001 - curated layer absent if exposure job failed
            print(f"[dq_job] curated {name} unavailable: {type(exc).__name__}")
    datasets["manifest"] = read_manifest(spark, settings, business_date)

    results = DQEngine(load_rules(), SparkBackend()).run(datasets, business_date, run_id=f"spark-dq-{business_date:%Y%m%d}")
    engine = get_engine(settings.database_url)
    ensure_schema(engine)
    write_results(engine, results, business_date)
    spark.stop()
    return {"business_date": business_date.isoformat(), "rules": len(results), "breaks": sum(r.is_break for r in results)}


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--business-date", type=date.fromisoformat, required=True)
    print(run(ap.parse_args(argv).business_date))


if __name__ == "__main__":
    main()
