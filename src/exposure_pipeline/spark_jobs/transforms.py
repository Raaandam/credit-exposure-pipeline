"""Spark DataFrame curation + aggregation. Mirrors ``transform/pandas_impl.py`` rule for rule.

No Python UDFs: lookups (rating band, CCF, rank) are ``create_map`` literals and the FX
table is broadcast, so all work stays in the JVM and executors need no project code.
"""
from __future__ import annotations

from datetime import date
from itertools import chain

from pyspark.sql import Column, DataFrame, Window
from pyspark.sql import functions as F

from exposure_pipeline import reference as ref


def _map(d: dict) -> Column:
    return F.create_map(*[F.lit(x) for x in chain.from_iterable(d.items())])


def _norm(c: str) -> Column:
    return F.upper(F.trim(F.col(c)))


def _num(c: str) -> Column:
    return F.col(c).cast("double")


def _dedupe(df: DataFrame, keys: list[str]) -> DataFrame:
    """Keep the first occurrence per key in file order (matches pandas keep='first')."""
    w = Window.partitionBy(*keys).orderBy(F.col("_row_order"))
    return (df.withColumn("_row_order", F.monotonically_increasing_id())
            .withColumn("_rn", F.row_number().over(w)).filter("_rn = 1").drop("_rn", "_row_order"))


def _bd(business_date: date) -> Column:
    return F.lit(business_date.isoformat()).cast("date").alias("business_date")


def curate_fx(raw: DataFrame, business_date: date) -> DataFrame:
    return (_dedupe(raw.withColumn("currency", _norm("currency")), ["currency"])
            .select(_bd(business_date), "currency", _num("rate_to_chf").alias("rate_to_chf"))
            .dropna(subset=["currency", "rate_to_chf"]))


def curate_counterparties(raw: DataFrame, business_date: date) -> DataFrame:
    df = _dedupe(raw, ["counterparty_id"])
    rating = F.when(_norm("internal_rating").isin(list(ref.RATING_SCALE)), _norm("internal_rating"))
    sector = F.when(_norm("sector").isin(list(ref.SECTORS)), _norm("sector")).otherwise(F.lit(ref.UNMAPPED))
    return df.select(
        _bd(business_date), "counterparty_id", "counterparty_name", "lei", "group_id", "ultimate_parent_id",
        _norm("country").alias("country"), sector.alias("sector"), rating.alias("internal_rating"),
        F.coalesce(_map(ref.RATING_BAND)[rating], F.lit(ref.UNRATED_BAND)).alias("rating_band"),
    )


def curate_facilities(raw: DataFrame, counterparties: DataFrame, fx: DataFrame, business_date: date) -> DataFrame:
    df = _dedupe(raw, ["facility_id"])
    limit, drawn = _num("limit_amount"), _num("drawn_amount")
    df = (df.select(
            _bd(business_date), "facility_id", "counterparty_id", _norm("facility_type").alias("facility_type"),
            _norm("currency").alias("currency"), "source_system", limit.alias("limit_amount"),
            drawn.alias("drawn_amount"), "maturity_date")
          .join(F.broadcast(fx.select("currency", F.col("rate_to_chf").alias("fx_rate_to_chf"))), "currency", "left"))
    undrawn = F.greatest(F.col("limit_amount") - F.col("drawn_amount"), F.lit(0.0))
    df = (df.withColumn("limit_chf", F.col("limit_amount") * F.col("fx_rate_to_chf"))
            .withColumn("drawn_chf", F.col("drawn_amount") * F.col("fx_rate_to_chf"))
            .withColumn("undrawn_chf", undrawn * F.col("fx_rate_to_chf"))
            .withColumn("ccf", _map(ref.FACILITY_CCF)[F.col("facility_type")])
            .withColumn("ead_chf", F.col("drawn_chf") + F.col("ccf") * F.col("undrawn_chf")))
    cp = counterparties.select("counterparty_id", F.col("group_id").alias("_g"), F.col("sector").alias("_s"),
                               F.col("rating_band").alias("_r"), F.lit(True).alias("_known"))
    df = df.join(cp, "counterparty_id", "left")
    return (df.withColumn("is_orphan", F.col("_known").isNull())
              .withColumn("group_id", F.coalesce("_g", F.lit(ref.UNMAPPED)))
              .withColumn("sector", F.coalesce("_s", F.lit(ref.UNMAPPED)))
              .withColumn("rating_band", F.coalesce("_r", F.lit(ref.UNMAPPED)))
              .drop("_g", "_s", "_r", "_known"))


def curate_collateral(raw: DataFrame, fx: DataFrame, business_date: date) -> DataFrame:
    df = _dedupe(raw, ["collateral_id"])
    haircut = F.least(F.greatest(_num("haircut"), F.lit(0.0)), F.lit(1.0))
    return (df.select(_bd(business_date), "collateral_id", "facility_id", _norm("collateral_type").alias("collateral_type"),
                      _norm("currency").alias("currency"), _num("market_value").alias("market_value"),
                      _num("haircut").alias("haircut"), haircut.alias("_h"))
              .join(F.broadcast(fx.select("currency", "rate_to_chf")), "currency", "left")
              .withColumn("adjusted_value_chf", F.col("market_value") * (1 - F.col("_h")) * F.col("rate_to_chf"))
              .drop("_h", "rate_to_chf"))


def curate_rating_changes(raw: DataFrame, business_date: date) -> DataFrame:
    df = _dedupe(raw, ["counterparty_id"])
    rank = _map(ref.RATING_RANK)
    notch = (rank[_norm("new_rating")] - rank[_norm("previous_rating")]).cast("double")
    return df.select(_bd(business_date), "counterparty_id", _norm("previous_rating").alias("previous_rating"),
                     _norm("new_rating").alias("new_rating"), notch.alias("notch_change"),
                     F.abs(notch).alias("notch_change_abs"), "effective_date", "change_reason")


def build_facility_exposure(facilities: DataFrame, collateral: DataFrame) -> DataFrame:
    coll = collateral.groupBy("facility_id").agg(F.sum("adjusted_value_chf").alias("_coll"))
    fe = facilities.join(coll, "facility_id", "left")
    # F.least ignores NULLs, so guard explicitly: no EAD -> no recognised collateral (as in pandas)
    capped = F.when(F.col("ead_chf").isNull(), F.lit(None).cast("double")).otherwise(
        F.least(F.coalesce(F.col("_coll"), F.lit(0.0)), F.col("ead_chf")))
    return (fe.withColumn("collateral_chf", capped)
              .withColumn("net_exposure_chf", F.col("ead_chf") - F.col("collateral_chf"))
              .drop("_coll"))


def aggregate_exposure(fe: DataFrame, business_date: date) -> DataFrame:
    frames = []
    for dim, column in ref.AGGREGATION_DIMENSIONS.items():
        frames.append(
            fe.groupBy(F.col(column).alias("dimension_value"))
              .agg(F.sum("ead_chf").alias("ead_chf"), F.sum("collateral_chf").alias("collateral_chf"),
                   F.sum("net_exposure_chf").alias("net_exposure_chf"), F.count("facility_id").alias("facility_count"),
                   F.countDistinct("counterparty_id").alias("counterparty_count"))
              .select(_bd(business_date), F.lit(dim).alias("dimension"), "dimension_value", "ead_chf", "collateral_chf",
                      "net_exposure_chf", "facility_count", "counterparty_count"))
    out = frames[0]
    for f in frames[1:]:
        out = out.unionByName(f)
    return out


def curate_all(raw: dict[str, DataFrame], business_date: date) -> dict[str, DataFrame]:
    fx = curate_fx(raw["raw.fx_rates"], business_date).cache()
    cps = curate_counterparties(raw["raw.counterparties"], business_date).cache()
    fac = curate_facilities(raw["raw.facilities"], cps, fx, business_date)
    col = curate_collateral(raw["raw.collateral"], fx, business_date)
    rc = curate_rating_changes(raw["raw.rating_changes"], business_date)
    fe = build_facility_exposure(fac, col).cache()
    return {
        "curated.fx_rate": fx,
        "curated.counterparty": cps,
        "curated.facility_exposure": fe,
        "curated.collateral": col,
        "curated.rating_change": rc,
        "mart.exposure_aggregate": aggregate_exposure(fe, business_date),
    }
