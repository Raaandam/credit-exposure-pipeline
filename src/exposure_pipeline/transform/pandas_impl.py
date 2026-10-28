"""pandas implementation of curation and exposure aggregation.

Business rules (kept identical in the Spark version):
* Raw feeds are read as strings (schema-on-read) so DQ rules can see bad values; casting
  happens here with ``errors="coerce"`` - unparseable values become NULL and are caught by
  the completeness/validity rules rather than crashing the load.
* Duplicates on the business key are removed keeping the first occurrence (file order).
* EAD (CHF) = drawn + CCF x max(limit - drawn, 0), converted at the day's FX rate.
* Collateral value = market value x (1 - haircut) x FX; haircut clipped to [0, 1] so a
  percent-vs-decimal error makes collateral worthless (conservative) instead of negative.
* Collateral recognised per facility is capped at EAD; net exposure = EAD - collateral.
* Facilities whose counterparty is unknown are kept and reported in an UNMAPPED bucket so
  totals still reconcile; unrated counterparties go to the UNRATED band.
"""
from __future__ import annotations

from datetime import date

import numpy as np
import pandas as pd

from exposure_pipeline import reference as ref


def _num(s: pd.Series) -> pd.Series:
    return pd.to_numeric(s, errors="coerce")


def _norm(s: pd.Series) -> pd.Series:
    """Trim + upper-case, preserving missing values."""
    return s.where(s.isna(), s.astype(str).str.strip().str.upper())


def curate_fx(raw: pd.DataFrame, business_date: date) -> pd.DataFrame:
    df = raw.copy()
    df["currency"] = _norm(df["currency"])
    df = df.drop_duplicates("currency", keep="first")
    return pd.DataFrame(
        {"business_date": business_date, "currency": df["currency"], "rate_to_chf": _num(df["rate_to_chf"])}
    ).dropna(subset=["currency", "rate_to_chf"])


def curate_counterparties(raw: pd.DataFrame, business_date: date) -> pd.DataFrame:
    df = raw.drop_duplicates("counterparty_id", keep="first")
    rating = _norm(df["internal_rating"])
    rating = rating.where(rating.isin(ref.RATING_SCALE))
    sector = _norm(df["sector"])
    sector = sector.where(sector.isin(ref.SECTORS), ref.UNMAPPED)
    return pd.DataFrame(
        {
            "business_date": business_date,
            "counterparty_id": df["counterparty_id"],
            "counterparty_name": df["counterparty_name"],
            "lei": df["lei"],
            "group_id": df["group_id"],
            "ultimate_parent_id": df["ultimate_parent_id"],
            "country": _norm(df["country"]),
            "sector": sector,
            "internal_rating": rating,
            "rating_band": rating.map(ref.RATING_BAND).fillna(ref.UNRATED_BAND),
        }
    ).reset_index(drop=True)


def curate_facilities(raw: pd.DataFrame, counterparties: pd.DataFrame, fx: pd.DataFrame, business_date: date) -> pd.DataFrame:
    df = raw.drop_duplicates("facility_id", keep="first")
    fx_map = dict(zip(fx["currency"], fx["rate_to_chf"]))
    ccy = _norm(df["currency"])
    ftype = _norm(df["facility_type"])
    limit, drawn = _num(df["limit_amount"]), _num(df["drawn_amount"])
    rate = ccy.map(fx_map)
    ccf = ftype.map(ref.FACILITY_CCF)
    undrawn = (limit - drawn).clip(lower=0)

    out = pd.DataFrame(
        {
            "business_date": business_date,
            "facility_id": df["facility_id"],
            "counterparty_id": df["counterparty_id"],
            "facility_type": ftype,
            "currency": ccy,
            "source_system": df["source_system"],
            "limit_amount": limit,
            "drawn_amount": drawn,
            "fx_rate_to_chf": rate,
            "limit_chf": limit * rate,
            "drawn_chf": drawn * rate,
            "undrawn_chf": undrawn * rate,
            "ccf": ccf,
            "maturity_date": df["maturity_date"],
        }
    )
    out["ead_chf"] = out["drawn_chf"] + out["ccf"] * out["undrawn_chf"]

    cp = counterparties[["counterparty_id", "group_id", "sector", "rating_band"]]
    out = out.merge(cp, on="counterparty_id", how="left", indicator=True)
    out["is_orphan"] = out.pop("_merge").eq("left_only")
    out["group_id"] = out["group_id"].fillna(ref.UNMAPPED)
    out["sector"] = out["sector"].fillna(ref.UNMAPPED)
    out["rating_band"] = out["rating_band"].fillna(ref.UNMAPPED)
    return out


def curate_collateral(raw: pd.DataFrame, fx: pd.DataFrame, business_date: date) -> pd.DataFrame:
    df = raw.drop_duplicates("collateral_id", keep="first")
    fx_map = dict(zip(fx["currency"], fx["rate_to_chf"]))
    ccy = _norm(df["currency"])
    mv, haircut = _num(df["market_value"]), _num(df["haircut"])
    rate = ccy.map(fx_map)
    return pd.DataFrame(
        {
            "business_date": business_date,
            "collateral_id": df["collateral_id"],
            "facility_id": df["facility_id"],
            "collateral_type": _norm(df["collateral_type"]),
            "currency": ccy,
            "market_value": mv,
            "haircut": haircut,
            "adjusted_value_chf": mv * (1 - haircut.clip(0, 1)) * rate,
        }
    ).reset_index(drop=True)


def curate_rating_changes(raw: pd.DataFrame, business_date: date) -> pd.DataFrame:
    df = raw.drop_duplicates("counterparty_id", keep="first")
    prev, new = _norm(df["previous_rating"]), _norm(df["new_rating"])
    notch = new.map(ref.RATING_RANK) - prev.map(ref.RATING_RANK)  # positive = downgrade
    return pd.DataFrame(
        {
            "business_date": business_date,
            "counterparty_id": df["counterparty_id"],
            "previous_rating": prev,
            "new_rating": new,
            "notch_change": notch,
            "notch_change_abs": notch.abs(),
            "effective_date": df["effective_date"],
            "change_reason": df["change_reason"],
        }
    ).reset_index(drop=True)


def build_facility_exposure(facilities: pd.DataFrame, collateral: pd.DataFrame) -> pd.DataFrame:
    coll = (
        collateral.groupby("facility_id", as_index=False)["adjusted_value_chf"].sum()
        .rename(columns={"adjusted_value_chf": "collateral_value_chf"})
    )
    fe = facilities.merge(coll, on="facility_id", how="left")
    # np.minimum propagates NaN: no EAD (missing FX/limit) -> no collateral recognised either.
    fe["collateral_chf"] = np.minimum(fe["collateral_value_chf"].fillna(0.0), fe["ead_chf"])
    fe["net_exposure_chf"] = fe["ead_chf"] - fe["collateral_chf"]
    return fe.drop(columns=["collateral_value_chf"])


def aggregate_exposure(facility_exposure: pd.DataFrame, business_date: date) -> pd.DataFrame:
    """Long-format mart: one row per (dimension, dimension_value). One table serves the
    dashboard, limit monitoring (by group) and RWA input (by rating band), and makes the
    "every dimension sums to the same total" reconciliation a simple filter."""
    frames = []
    for dim, column in ref.AGGREGATION_DIMENSIONS.items():
        g = (
            facility_exposure.groupby(column, dropna=False)
            .agg(
                ead_chf=("ead_chf", "sum"),
                collateral_chf=("collateral_chf", "sum"),
                net_exposure_chf=("net_exposure_chf", "sum"),
                facility_count=("facility_id", "count"),
                counterparty_count=("counterparty_id", "nunique"),
            )
            .reset_index()
            .rename(columns={column: "dimension_value"})
        )
        g.insert(0, "dimension", dim)
        frames.append(g)
    out = pd.concat(frames, ignore_index=True)
    out.insert(0, "business_date", business_date)
    return out


def curate_all(raw: dict[str, pd.DataFrame], business_date: date) -> dict[str, pd.DataFrame]:
    """Run the full curation for one business date. Missing feeds yield empty outputs."""
    fx = curate_fx(raw["raw.fx_rates"], business_date) if "raw.fx_rates" in raw else pd.DataFrame(columns=["business_date", "currency", "rate_to_chf"])
    cps = curate_counterparties(raw["raw.counterparties"], business_date)
    fac = curate_facilities(raw["raw.facilities"], cps, fx, business_date)
    col = curate_collateral(raw["raw.collateral"], fx, business_date) if "raw.collateral" in raw else pd.DataFrame(
        columns=["business_date", "collateral_id", "facility_id", "adjusted_value_chf"])
    rc = curate_rating_changes(raw["raw.rating_changes"], business_date) if "raw.rating_changes" in raw else pd.DataFrame(
        columns=["business_date", "counterparty_id", "notch_change_abs"])
    fe = build_facility_exposure(fac, col)
    agg = aggregate_exposure(fe, business_date)
    return {
        "curated.fx_rate": fx,
        "curated.counterparty": cps,
        "curated.facility_exposure": fe,
        "curated.collateral": col,
        "curated.rating_change": rc,
        "mart.exposure_aggregate": agg,
    }
