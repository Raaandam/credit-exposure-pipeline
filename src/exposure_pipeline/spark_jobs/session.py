"""SparkSession factory, raw feed readers and JDBC helpers."""
from __future__ import annotations

import json
from datetime import date

from pyspark.sql import DataFrame, SparkSession
from pyspark.sql.types import StringType, StructField, StructType
from sqlalchemy.engine import make_url

from exposure_pipeline import reference as ref
from exposure_pipeline.config import Settings
from exposure_pipeline.storage.blob import raw_uri, spark_hadoop_conf

# Raw schemas: everything is a string on purpose (schema-on-read). A value like "N/A" in
# limit_amount must survive ingestion so the DQ engine can report it, instead of Spark
# silently nulling it under inferSchema. Explicit schemas also avoid a full inference pass.
RAW_COLUMNS: dict[str, list[str]] = {
    "counterparties": ["counterparty_id", "counterparty_name", "lei", "group_id", "ultimate_parent_id", "country",
                       "sector", "internal_rating", "as_of_date"],
    "facilities": ["facility_id", "counterparty_id", "facility_type", "currency", "limit_amount", "drawn_amount",
                   "maturity_date", "source_system", "as_of_date"],
    "collateral": ["collateral_id", "facility_id", "collateral_type", "currency", "market_value", "haircut", "as_of_date"],
    "rating_changes": ["counterparty_id", "previous_rating", "new_rating", "effective_date", "change_reason", "rating_source"],
    "fx_rates": ["currency", "rate_to_chf", "as_of_date"],
}


def raw_schema(feed: str) -> StructType:
    return StructType([StructField(c, StringType(), True) for c in RAW_COLUMNS[feed]])


def build_spark(app_name: str, settings: Settings) -> SparkSession:
    builder = (
        SparkSession.builder.appName(app_name)
        .master(settings.spark_master)
        .config("spark.jars.packages", settings.spark_jars_packages)
        # Overwrite only the partitions present in the written DataFrame -> rerunning one
        # business date never wipes other dates in the curated parquet layer.
        .config("spark.sql.sources.partitionOverwriteMode", "dynamic")
        .config("spark.sql.session.timeZone", "UTC")
        # Small daily volumes: avoid 200 tiny shuffle partitions.
        .config("spark.sql.shuffle.partitions", "16")
    )
    for key, value in spark_hadoop_conf(settings).items():
        builder = builder.config(f"spark.hadoop.{key}", value)
    return builder.getOrCreate()


def read_raw(spark: SparkSession, settings: Settings, business_date: date) -> dict[str, DataFrame]:
    """Read every raw feed for one business date. Missing files are skipped (DQ reports them)."""
    out: dict[str, DataFrame] = {}
    for feed, fname in ref.FEED_FILES.items():
        uri = raw_uri(settings, business_date, fname)
        try:
            if fname.endswith(".json"):
                df = spark.read.schema(raw_schema(feed)).json(uri)
            else:
                df = spark.read.schema(raw_schema(feed)).option("header", True).option("mode", "PERMISSIVE").csv(uri)
            df.limit(1).collect()  # force path resolution so a missing file is detected here
            out[f"raw.{feed}"] = df
        except Exception as exc:  # noqa: BLE001 - AnalysisException: path does not exist
            print(f"[read_raw] {feed} not available at {uri}: {type(exc).__name__}")
    return out


def read_manifest(spark: SparkSession, settings: Settings, business_date: date) -> dict:
    uri = raw_uri(settings, business_date, ref.MANIFEST_FILE)
    try:
        text = "\n".join(r.value for r in spark.read.text(uri).collect())
        return json.loads(text).get("feeds", {})
    except Exception as exc:  # noqa: BLE001
        print(f"[read_manifest] manifest not available at {uri}: {type(exc).__name__}")
        return {}


def jdbc_options(database_url: str) -> dict[str, str]:
    """Derive Spark JDBC options from the SQLAlchemy URL so there is one DB setting."""
    url = make_url(database_url)
    return {
        "url": f"jdbc:postgresql://{url.host}:{url.port or 5432}/{url.database}",
        "user": url.username or "",
        "password": url.password or "",
        "driver": "org.postgresql.Driver",
    }


def write_jdbc(df: DataFrame, table: str, database_url: str, mode: str = "overwrite") -> None:
    (df.write.format("jdbc").options(**jdbc_options(database_url)).option("dbtable", table)
     .option("batchsize", 10000).mode(mode).save())
