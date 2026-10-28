"""Daily exposure job: raw feeds (Azure Blob) -> curated parquet (Blob) + Postgres tables.

    spark-submit --master $SPARK_MASTER_URL --packages $SPARK_JARS_PACKAGES \
        src/exposure_pipeline/spark_jobs/daily_exposure_job.py --business-date 2026-01-07

Idempotency (why reruns/backfills are safe):
1. Curated parquet is written with partitionOverwriteMode=dynamic, partitioned by
   business_date -> only that date's folder is replaced.
2. Postgres: Spark JDBC writes the day into a staging table (mode=overwrite), then one
   transaction deletes the target partition and inserts from staging.
3. load_audit is replaced for (business_date) in the same way.
Running the job twice for a date therefore yields exactly the same tables.
"""
from __future__ import annotations

import argparse
import uuid
from datetime import date, datetime, timezone

import pandas as pd

from exposure_pipeline import reference as ref
from exposure_pipeline.config import get_settings
from exposure_pipeline.spark_jobs.session import build_spark, read_manifest, read_raw, write_jdbc
from exposure_pipeline.spark_jobs.transforms import curate_all
from exposure_pipeline.storage.blob import curated_uri, raw_uri
from exposure_pipeline.storage.db import ensure_schema, get_engine, replace_partition, swap_partition_from_staging
from exposure_pipeline.storage.tables import TABLES_BY_DATASET, load_audit

CORE_FEEDS = ("raw.counterparties", "raw.facilities", "raw.collateral", "raw.rating_changes", "raw.fx_rates")


def run(business_date: date) -> dict:
    settings = get_settings()
    spark = build_spark(f"credit-exposure-{business_date}", settings)
    engine = get_engine(settings.database_url)
    ensure_schema(engine)
    run_id = f"spark-{business_date:%Y%m%d}-{uuid.uuid4().hex[:8]}"

    raw = read_raw(spark, settings, business_date)
    manifest = read_manifest(spark, settings, business_date)
    missing = [f for f in CORE_FEEDS if f not in raw]

    # load_audit first, so a failed day still records which feeds arrived
    now = datetime.now(timezone.utc).replace(tzinfo=None)
    audit = pd.DataFrame([{
        "business_date": business_date, "feed": feed, "run_id": run_id,
        "status": "SUCCESS" if f"raw.{feed}" in raw else "MISSING",
        "row_count": raw[f"raw.{feed}"].count() if f"raw.{feed}" in raw else None,
        "source_path": raw_uri(settings, business_date, fname),
        "source_system": manifest.get(feed, {}).get("source_system"),
        "manifest_business_date": manifest.get(feed, {}).get("business_date"),
        "manifest_record_count": manifest.get(feed, {}).get("record_count"),
        "extract_timestamp": manifest.get(feed, {}).get("extract_timestamp"),
        "loaded_at": now,
    } for feed, fname in ref.FEED_FILES.items()])
    with engine.begin() as conn:
        replace_partition(conn, load_audit, audit, business_date)

    if missing:
        # Fail loudly; the DQ task still runs (trigger_rule=all_done) and records the gap.
        raise RuntimeError(f"core feeds missing for {business_date}: {missing}")

    curated = curate_all(raw, business_date)
    counts = {}
    for name, table in TABLES_BY_DATASET.items():
        df = curated[name]
        # 1) curated parquet layer in Blob/ADLS (consumed by the DQ job and downstream Spark users)
        df.write.mode("overwrite").partitionBy("business_date").parquet(curated_uri(settings, table.name))
        # 2) Postgres serving layer via staging + atomic swap
        staging = f"stg_{table.name}_{business_date:%Y%m%d}"
        cols = [c for c in table.columns.keys() if c in df.columns]
        write_jdbc(df.select(*cols), staging, settings.database_url)
        swap_partition_from_staging(engine, table.name, staging, business_date, cols)
        counts[table.name] = df.count()

    spark.stop()
    return {"business_date": business_date.isoformat(), "run_id": run_id, "rows": counts}


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--business-date", type=date.fromisoformat, required=True)
    print(run(ap.parse_args(argv).business_date))


if __name__ == "__main__":
    main()
