"""Synthetic, related credit-risk source feeds with labelled data errors.

Produces, per business date (Hive-style partitions under ``<out>/raw/business_date=YYYY-MM-DD``):

* ``counterparties.csv``  - counterparty master snapshot (group, ultimate parent, sector, rating)
* ``facilities.csv``      - credit facility snapshot (limit, drawn, currency, source system)
* ``collateral.json``     - collateral snapshot as newline-delimited JSON
* ``rating_changes.csv``  - daily internal rating migrations (delta feed)
* ``fx_rates.csv``        - FX rates to CHF
* ``manifest.json``       - source control file: record counts, control totals, extract time

and ``<out>/labels/error_labels.csv`` with the ground truth for every injected error.

Design notes:
* Day 0 is kept clean so there is always a "last successful load" to compare against.
* Record-level errors are injected *before* the manifest is computed (the source system
  itself sent bad rows, so its control file agrees with them). Feed-level incidents
  (truncated transfer, unit switch, stale copy, missing FX row) are applied *after*, which
  is what makes count/total reconciliation against the control file meaningful.
* Error types are deliberately overlapping per rule (e.g. a referential-integrity break can
  be an orphan id = mapping_error, or a truncated master file = missing_source_data), so
  triage is not a trivial rule-id -> category lookup.
"""
from __future__ import annotations

import argparse
import json
import logging
import math
from dataclasses import dataclass, field
from datetime import date, datetime, time, timedelta
from pathlib import Path

import numpy as np
import pandas as pd

from exposure_pipeline import reference as ref
from exposure_pipeline.generator.labels import LabelLog

log = logging.getLogger(__name__)

MISSING = "missing_source_data"
MAPPING = "mapping_error"
STALE = "stale_feed"
DUPLICATE = "duplicate"
UNIT_FX = "unit_or_fx_mismatch"
GENUINE = "genuine_business_change"

ADJECTIVES = [
    "Alpine", "Nordic", "Atlas", "Helvetic", "Pacific", "Meridian", "Crescent", "Summit", "Harbour",
    "Granite", "Silver", "Orion", "Cedar", "Falcon", "Lakeside", "Northern", "Eastern", "Vertex",
]
NOUNS = [
    "Holdings", "Capital", "Industries", "Energy", "Logistics", "Properties", "Partners", "Pharma",
    "Systems", "Foods", "Shipping", "Utilities", "Insurance", "Securities", "Motors", "Telecom",
]
SUFFIXES = ["AG", "SA", "GmbH", "Ltd", "plc", "Inc", "LLC", "NV", "SpA", "Pte Ltd", "KK", "BV"]

COUNTRY_WEIGHTS = {
    "CH": 0.18, "US": 0.20, "GB": 0.12, "DE": 0.10, "FR": 0.08, "NL": 0.04, "IT": 0.04,
    "ES": 0.03, "JP": 0.06, "SG": 0.05, "HK": 0.05, "CA": 0.03, "AU": 0.02,
}
COUNTRY_CCY = {
    "CH": "CHF", "US": "USD", "GB": "GBP", "DE": "EUR", "FR": "EUR", "NL": "EUR", "IT": "EUR",
    "ES": "EUR", "JP": "JPY", "SG": "SGD", "HK": "HKD", "CA": "CAD", "AU": "AUD",
}
APAC = {"JP", "SG", "HK", "AU"}
AMER = {"US", "CA"}
RATING_DIST = [0.03, 0.08, 0.17, 0.30, 0.22, 0.13, 0.05, 0.02]
FACILITY_TYPE_WEIGHTS = {"TERM_LOAN": 0.35, "REVOLVER": 0.30, "TRADE_FINANCE": 0.12, "GUARANTEE": 0.13, "OVERDRAFT": 0.10}
COLLATERAL_TYPE_WEIGHTS = {"CASH": 0.12, "GOVT_BOND": 0.15, "CORP_BOND": 0.13, "EQUITY": 0.15, "REAL_ESTATE": 0.30, "RECEIVABLES": 0.15}
FEED_SOURCE_SYSTEM = {
    "counterparties": "CPTY_MASTER",
    "facilities": "LOAN_BOOKING",
    "collateral": "COLLATERAL_MGMT",
    "rating_changes": "INTERNAL_RATING_SYS",
    "fx_rates": "MARKET_DATA",
}
FEED_INCIDENTS = [
    "facilities_unit_thousands",
    "rating_changes_stale",
    "facilities_truncation",
    "collateral_stale",
    "counterparties_truncation",
    "fx_missing_currency",
]


@dataclass
class GeneratorConfig:
    out_dir: Path
    n_counterparties: int = 5000
    n_days: int = 10
    start_date: date = date(2026, 1, 5)
    seed: int = 42
    error_scale: float = 1.0
    # Optional explicit incident per day (None = clean). Defaults to a seeded shuffle.
    incident_schedule: list[str | None] | None = None
    # Probability that a given record-level error type occurs on a given day.
    error_type_probability: float = 0.75
    stats: dict = field(default_factory=dict)


def business_days(start: date, n: int) -> list[date]:
    days, d = [], start
    while len(days) < n:
        if d.weekday() < 5:
            days.append(d)
        d += timedelta(days=1)
    return days


# --------------------------------------------------------------------------- universe
def _build_counterparties(n: int, rng: np.random.Generator) -> pd.DataFrame:
    n_groups = max(1, n // 4)
    group_idx = np.concatenate([np.arange(n_groups), rng.integers(0, n_groups, max(0, n - n_groups))])[:n]
    rng.shuffle(group_idx)
    ids = np.array([f"CP{i + 1:06d}" for i in range(n)])
    df = pd.DataFrame({"counterparty_id": ids, "_g": group_idx})
    # Ultimate parent = lowest-numbered counterparty of the group (zero-padded ids sort correctly).
    parent_of_group = df.groupby("_g")["counterparty_id"].min()

    countries = rng.choice(list(COUNTRY_WEIGHTS), size=n, p=list(COUNTRY_WEIGHTS.values()))
    group_sector = rng.choice(ref.SECTORS, size=n_groups)
    own_sector = rng.choice(ref.SECTORS, size=n)
    sector = np.where(rng.random(n) < 0.7, group_sector[group_idx], own_sector)
    names = [
        f"{a} {b} {s}"
        for a, b, s in zip(rng.choice(ADJECTIVES, n), rng.choice(NOUNS, n), rng.choice(SUFFIXES, n))
    ]
    alnum = np.array(list("ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789"))
    lei_body = rng.choice(alnum, size=(n, 18))
    lei = ["".join(row) + f"{chk:02d}" for row, chk in zip(lei_body, rng.integers(10, 100, n))]

    return pd.DataFrame(
        {
            "counterparty_id": ids,
            "counterparty_name": names,
            "lei": lei,
            "group_id": [f"GRP{g + 1:05d}" for g in group_idx],
            "ultimate_parent_id": df["_g"].map(parent_of_group).to_numpy(),
            "country": countries,
            "sector": sector,
            "internal_rating": rng.choice(ref.RATING_SCALE, size=n, p=RATING_DIST),
        }
    )


def _build_facilities(cps: pd.DataFrame, start: date, rng: np.random.Generator) -> pd.DataFrame:
    n_per = 1 + rng.poisson(1.5, len(cps))
    cp_idx = np.repeat(np.arange(len(cps)), n_per)
    n = len(cp_idx)
    ftype = rng.choice(list(FACILITY_TYPE_WEIGHTS), size=n, p=list(FACILITY_TYPE_WEIGHTS.values()))
    country = cps["country"].to_numpy()[cp_idx]
    ccy = np.where(rng.random(n) < 0.2, "USD", pd.Series(country).map(COUNTRY_CCY).to_numpy())
    limit = np.clip(rng.lognormal(np.log(2e6), 1.3, n), 5e4, 1e9).round(-3)

    util = np.empty(n)
    for t, (a, b) in {
        "TERM_LOAN": (18, 1.2), "REVOLVER": (2, 3), "TRADE_FINANCE": (2, 2), "GUARANTEE": (5, 2), "OVERDRAFT": (1, 4),
    }.items():
        mask = ftype == t
        util[mask] = rng.beta(a, b, mask.sum())
    drawn = np.round(limit * util, 2)

    source = np.where(np.isin(country, list(APAC)), "LOAN_SYS_APAC", np.where(np.isin(country, list(AMER)), "LOAN_SYS_AMER", "LOAN_SYS_EMEA"))
    source = np.where(ftype == "TRADE_FINANCE", "TRADE_FIN_SYS", source)
    maturity = [(start + timedelta(days=int(d))).isoformat() for d in rng.integers(30, 3650, n)]

    return pd.DataFrame(
        {
            "facility_id": [f"FAC{j + 1:07d}" for j in range(n)],
            "counterparty_id": cps["counterparty_id"].to_numpy()[cp_idx],
            "facility_type": ftype,
            "currency": ccy,
            "limit_amount": limit,
            "drawn_amount": drawn,
            "maturity_date": maturity,
            "source_system": source,
        }
    )


def _build_collateral(fac: pd.DataFrame, rng: np.random.Generator) -> pd.DataFrame:
    has = np.flatnonzero(rng.random(len(fac)) < 0.55)
    count = 1 + (rng.random(len(has)) < 0.3).astype(int)
    fac_rep = np.repeat(has, count)
    cnt_rep = np.repeat(count, count)
    n = len(fac_rep)
    ctype = rng.choice(list(COLLATERAL_TYPE_WEIGHTS), size=n, p=list(COLLATERAL_TYPE_WEIGHTS.values()))
    base_haircut = pd.Series(ctype).map(ref.COLLATERAL_HAIRCUT).to_numpy()
    haircut = np.clip(base_haircut + rng.uniform(-0.03, 0.05, n), 0, 0.95).round(4)
    mv = (fac["limit_amount"].to_numpy()[fac_rep] * rng.uniform(0.2, 1.1, n) / cnt_rep).round(2)
    fac_ccy = fac["currency"].to_numpy()[fac_rep]
    ccy = np.where(rng.random(n) < 0.1, rng.choice(["CHF", "USD"], n), fac_ccy)
    return pd.DataFrame(
        {
            "collateral_id": [f"COL{k + 1:07d}" for k in range(n)],
            "facility_id": fac["facility_id"].to_numpy()[fac_rep],
            "collateral_type": ctype,
            "currency": ccy,
            "market_value": mv,
            "haircut": haircut,
        }
    )


# --------------------------------------------------------------------------- daily evolution
def _drift(fac: pd.DataFrame, col: pd.DataFrame, fx: dict[str, float], rng: np.random.Generator) -> None:
    factor = 1 + rng.normal(0, 0.01, len(fac))
    fac["drawn_amount"] = np.minimum(np.maximum(fac["drawn_amount"] * factor, 0), fac["limit_amount"]).round(2)
    col["market_value"] = (col["market_value"] * (1 + rng.normal(0, 0.005, len(col)))).round(2)
    for c in fx:
        if c != "CHF":
            fx[c] = round(fx[c] * (1 + rng.normal(0, 0.003)), 6)


def _rating_migrations(cps: pd.DataFrame, bd: date, rng: np.random.Generator, genuine: bool) -> tuple[pd.DataFrame, list[str]]:
    """Apply today's rating migrations to the counterparty state and return the delta feed.

    Normal migrations are one notch. On most days a handful of *genuine* multi-notch
    downgrades (credit events) are added - these legitimately trip the "multi-notch
    migration" rule and are labelled genuine_business_change.
    """
    n = len(cps)
    ranks = cps["internal_rating"].map(ref.RATING_RANK).to_numpy()
    n_normal = max(1, int(n * 0.004))
    n_genuine = int(rng.integers(1, 4)) if genuine else 0
    genuine_pool = np.flatnonzero(ranks <= 4)
    g_idx = rng.choice(genuine_pool, size=min(n_genuine, len(genuine_pool)), replace=False) if n_genuine else np.array([], int)
    normal_pool = np.setdiff1d(np.arange(n), g_idx)
    m_idx = rng.choice(normal_pool, size=n_normal, replace=False)

    new_ranks = ranks.copy()
    step = rng.choice([-1, 1], size=len(m_idx))
    proposed = ranks[m_idx] + step
    proposed = np.where((proposed < 0) | (proposed > len(ref.RATING_SCALE) - 1), ranks[m_idx] - step, proposed)
    new_ranks[m_idx] = proposed
    if len(g_idx):
        new_ranks[g_idx] = np.minimum(ranks[g_idx] + rng.integers(3, 5, len(g_idx)), len(ref.RATING_SCALE) - 1)

    idx = np.concatenate([m_idx, g_idx])
    reasons = np.concatenate([rng.choice(["ANNUAL_REVIEW", "FINANCIALS_UPDATE", "WATCHLIST_REVIEW"], len(m_idx)), np.repeat("CREDIT_EVENT", len(g_idx))])
    delta = pd.DataFrame(
        {
            "counterparty_id": cps["counterparty_id"].to_numpy()[idx],
            "previous_rating": [ref.RATING_SCALE[r] for r in ranks[idx]],
            "new_rating": [ref.RATING_SCALE[r] for r in new_ranks[idx]],
            "effective_date": bd.isoformat(),
            "change_reason": reasons,
            "rating_source": "INTERNAL",
        }
    )
    cps["internal_rating"] = [ref.RATING_SCALE[r] for r in new_ranks]
    return delta, list(cps["counterparty_id"].to_numpy()[g_idx])


# --------------------------------------------------------------------------- record-level errors
class _Injector:
    """Picks disjoint row sets per feed so each record carries at most one injected error."""

    def __init__(self, cfg: GeneratorConfig, rng: np.random.Generator, labels: LabelLog, bd: date):
        self.cfg, self.rng, self.labels, self.bd = cfg, rng, labels, bd
        self.used: dict[str, set[int]] = {}

    def active(self) -> bool:
        return self.rng.random() < self.cfg.error_type_probability

    def pick(self, feed: str, df: pd.DataFrame, rate: float, candidates: np.ndarray | None = None) -> np.ndarray:
        used = self.used.setdefault(feed, set())
        pool = np.arange(len(df)) if candidates is None else candidates
        pool = np.array([i for i in pool if i not in used], dtype=int)
        if len(pool) == 0:
            return pool
        k = max(1, int(round(rate * len(df) * self.cfg.error_scale * self.rng.uniform(0.5, 1.5))))
        chosen = self.rng.choice(pool, size=min(k, len(pool)), replace=False)
        used.update(int(i) for i in chosen)
        return chosen

    def label(self, feed: str, df: pd.DataFrame, pos: np.ndarray, error_type: str, injection: str, column: str, desc: str) -> None:
        key = ref.FEED_KEYS[feed][0]
        self.labels.record(self.bd, feed, df[key].iloc[pos], error_type, injection, column, desc)


def _set(df: pd.DataFrame, pos: np.ndarray, column: str, values) -> None:
    df.loc[df.index[pos], column] = values


def _inject_counterparties(df: pd.DataFrame, inj: _Injector) -> pd.DataFrame:
    rng, f = inj.rng, "counterparties"
    if inj.active():
        p = inj.pick(f, df, 0.004)
        _set(df, p, "internal_rating", np.nan)
        inj.label(f, df, p, MISSING, "missing_rating", "internal_rating", "Rating not supplied by rating system")
    if inj.active():
        p = inj.pick(f, df, 0.002)
        _set(df, p, "sector", np.nan)
        inj.label(f, df, p, MISSING, "missing_sector", "sector", "Sector not populated at onboarding")
    if inj.active():
        p = inj.pick(f, df, 0.002)
        _set(df, p, "sector", rng.choice(["FIN-01", "RE_COMM", "IND_MFG"], len(p)))
        inj.label(f, df, p, MAPPING, "legacy_sector_code", "sector", "Legacy sector code not mapped to reporting taxonomy")
    if inj.active():
        p = inj.pick(f, df, 0.001)
        _set(df, p, "internal_rating", rng.choice(["Baa2", "Ba1", "A3"], len(p)))
        inj.label(f, df, p, MAPPING, "external_rating_code", "internal_rating", "Agency rating code loaded instead of internal scale")
    if inj.active():
        p = inj.pick(f, df, 0.001)
        _set(df, p, "country", rng.choice(["CHE", "USA", "Deutschland"], len(p)))
        inj.label(f, df, p, MAPPING, "non_iso_country", "country", "Country not in ISO 3166 alpha-2")
    if inj.active():
        p = inj.pick(f, df, 0.001)
        inj.label(f, df, p, DUPLICATE, "duplicate_row", "counterparty_id", "Counterparty sent twice in snapshot")
        df = pd.concat([df, df.iloc[p]], ignore_index=True)
    return df


def _inject_facilities(df: pd.DataFrame, inj: _Injector) -> pd.DataFrame:
    rng, f = inj.rng, "facilities"
    if inj.active():
        p = inj.pick(f, df, 0.003)
        _set(df, p, "counterparty_id", [f"CP9{x:05d}" for x in rng.integers(0, 100000, len(p))])
        inj.label(f, df, p, MAPPING, "orphan_counterparty", "counterparty_id", "Facility booked against counterparty id unknown to the master")
    if inj.active():
        gbp = np.flatnonzero(df["currency"].to_numpy() == "GBP")
        p = inj.pick(f, df, 0.001, candidates=gbp if len(gbp) else None)
        _set(df, p, "currency", "GBX")
        df.loc[df.index[p], ["limit_amount", "drawn_amount"]] = df.loc[df.index[p], ["limit_amount", "drawn_amount"]].to_numpy() * 100
        inj.label(f, df, p, UNIT_FX, "gbx_pence", "currency", "Amounts sent in pence (GBX) instead of GBP")
    if inj.active():
        p = inj.pick(f, df, 0.001)
        _set(df, p, "currency", np.nan)
        inj.label(f, df, p, MISSING, "missing_currency", "currency", "Currency not populated")
    if inj.active():
        p = inj.pick(f, df, 0.001)
        _set(df, p, "limit_amount", np.nan)
        inj.label(f, df, p, MISSING, "missing_limit", "limit_amount", "Limit not populated")
    if inj.active():
        p = inj.pick(f, df, 0.002)
        inj.label(f, df, p, DUPLICATE, "duplicate_row", "facility_id", "Facility extracted twice (overlapping extract windows)")
        dup = df.iloc[p].copy()
        dup["drawn_amount"] = np.minimum(dup["drawn_amount"] * 1.01, dup["limit_amount"]).round(2)  # near-duplicate
        df = pd.concat([df, dup], ignore_index=True)
    return df


def _inject_collateral(df: pd.DataFrame, inj: _Injector) -> pd.DataFrame:
    rng, f = inj.rng, "collateral"
    if inj.active():
        p = inj.pick(f, df, 0.003)
        _set(df, p, "market_value", np.nan)
        inj.label(f, df, p, MISSING, "missing_market_value", "market_value", "Valuation not available")
    if inj.active():
        p = inj.pick(f, df, 0.002)
        _set(df, p, "facility_id", [f"FAC9{x:06d}" for x in rng.integers(0, 1_000_000, len(p))])
        inj.label(f, df, p, MAPPING, "orphan_facility", "facility_id", "Collateral linked to facility id unknown to loan system")
    if inj.active():
        p = inj.pick(f, df, 0.002)
        df.loc[df.index[p], "haircut"] = (df.loc[df.index[p], "haircut"].to_numpy() * 100).round(2)
        inj.label(f, df, p, UNIT_FX, "haircut_in_percent", "haircut", "Haircut sent as percentage (25) instead of decimal (0.25)")
    if inj.active():
        p = inj.pick(f, df, 0.001)
        inj.label(f, df, p, DUPLICATE, "duplicate_row", "collateral_id", "Collateral record duplicated")
        df = pd.concat([df, df.iloc[p]], ignore_index=True)
    return df


def _inject_rating_changes(df: pd.DataFrame, inj: _Injector, genuine_ids: list[str]) -> pd.DataFrame:
    f = "rating_changes"
    if len(df) == 0:
        return df
    genuine_pos = np.flatnonzero(df["counterparty_id"].isin(genuine_ids).to_numpy())
    inj.used.setdefault(f, set()).update(int(i) for i in genuine_pos)
    inj.labels.record(inj.bd, f, df["counterparty_id"].iloc[genuine_pos], GENUINE, "credit_event", "new_rating", "Genuine multi-notch downgrade after credit event")
    if inj.rng.random() < 0.4:
        p = inj.pick(f, df, 0.0)
        _set(df, p, "new_rating", np.nan)
        inj.label(f, df, p, MISSING, "missing_new_rating", "new_rating", "Rating change without new grade")
    if inj.rng.random() < 0.4:
        p = inj.pick(f, df, 0.0)
        _set(df, p, "counterparty_id", [f"CP9{x:05d}" for x in inj.rng.integers(0, 100000, len(p))])
        inj.label(f, df, p, MAPPING, "orphan_counterparty", "counterparty_id", "Rating change for counterparty unknown to master")
    return df


# --------------------------------------------------------------------------- manifest + incidents
def _manifest(day: dict[str, pd.DataFrame], bd: date, rng: np.random.Generator) -> dict:
    extract_ts = datetime.combine(bd, time(22, 0)) + timedelta(minutes=int(rng.integers(0, 90)))
    feeds = {}
    for feed, df in day.items():
        entry = {
            "file": ref.FEED_FILES[feed],
            "source_system": FEED_SOURCE_SYSTEM[feed],
            "business_date": bd.isoformat(),
            "extract_timestamp": extract_ts.isoformat(),
            "record_count": int(len(df)),
        }
        if feed == "facilities":
            entry["control_total_limit"] = round(float(np.nansum(df["limit_amount"].to_numpy(dtype=float))), 2)
            entry["control_total_drawn"] = round(float(np.nansum(df["drawn_amount"].to_numpy(dtype=float))), 2)
        if feed == "collateral":
            entry["control_total_market_value"] = round(float(np.nansum(df["market_value"].to_numpy(dtype=float))), 2)
        feeds[feed] = entry
    return {"business_date": bd.isoformat(), "feeds": feeds}


def _apply_incident(name: str, day: dict, manifest: dict, prev: dict | None, bd: date, labels: LabelLog) -> bool:
    """Apply a whole-file incident after the source control file was produced."""
    if name == "facilities_unit_thousands":
        fac = day["facilities"]
        labelled = labels.keys(bd, "facilities")
        mask = (fac["source_system"] == "LOAN_SYS_APAC") & ~fac["facility_id"].isin(labelled)
        fac.loc[mask, ["limit_amount", "drawn_amount"]] = (fac.loc[mask, ["limit_amount", "drawn_amount"]] / 1000).round(2)
        labels.record(bd, "facilities", fac.loc[mask, "facility_id"], UNIT_FX, "unit_thousands", "limit_amount",
                      "APAC loan system switched to reporting amounts in thousands")
        labels.feed(bd, "facilities", UNIT_FX, name, "APAC loan system amounts in thousands; control totals in units")
        return True

    if name in {"rating_changes_stale", "collateral_stale"}:
        feed = name.replace("_stale", "")
        if prev is None:
            return False
        day[feed] = prev["frames"][feed].copy()
        manifest["feeds"][feed] = dict(prev["manifest"]["feeds"][feed])
        labels.drop(bd, feed)  # today's record-level injections never reached the file
        labels.feed(bd, feed, STALE, name, f"Previous day's {feed} file re-delivered (upstream job did not run)")
        return True

    if name == "facilities_truncation":
        fac = day["facilities"]
        n_drop = math.ceil(0.03 * len(fac))
        dropped = set(fac["facility_id"].iloc[-n_drop:]) - set(fac["facility_id"].iloc[:-n_drop])
        day["facilities"] = fac.iloc[:-n_drop].reset_index(drop=True)
        labels.feed(bd, "facilities", MISSING, name, "Facility file truncated in transfer (tail rows missing)")
        col = day["collateral"]
        hit = col["facility_id"].isin(dropped) & ~col["collateral_id"].isin(labels.keys(bd, "collateral"))
        labels.record(bd, "collateral", col.loc[hit, "collateral_id"], MISSING, name, "facility_id",
                      "Parent facility missing because facility file was truncated")
        return True

    if name == "counterparties_truncation":
        cps = day["counterparties"]
        n_drop = math.ceil(0.02 * len(cps))
        dropped = set(cps["counterparty_id"].iloc[-n_drop:]) - set(cps["counterparty_id"].iloc[:-n_drop])
        cps = cps.iloc[:-n_drop].reset_index(drop=True)
        day["counterparties"] = cps
        labels.feed(bd, "counterparties", MISSING, name, "Counterparty master file truncated in transfer")
        desc = "Counterparty missing because master file was truncated"
        for feed, key, fk in [
            ("facilities", "facility_id", "counterparty_id"),
            ("rating_changes", "counterparty_id", "counterparty_id"),
            ("counterparties", "counterparty_id", "ultimate_parent_id"),
        ]:
            df = day[feed]
            hit = df[fk].isin(dropped) & ~df[key].isin(labels.keys(bd, feed))
            labels.record(bd, feed, df.loc[hit, key], MISSING, name, fk, desc)
        return True

    if name == "fx_missing_currency":
        fx = day["fx_rates"]
        day["fx_rates"] = fx[fx["currency"] != "JPY"].reset_index(drop=True)
        labels.feed(bd, "fx_rates", MISSING, name, "JPY rate missing from market data snapshot")
        for feed, key in [("facilities", "facility_id"), ("collateral", "collateral_id")]:
            df = day[feed]
            hit = (df["currency"] == "JPY") & ~df[key].isin(labels.keys(bd, feed))
            labels.record(bd, feed, df.loc[hit, key], MISSING, name, "currency", "FX rate for currency not supplied")
        return True

    raise ValueError(f"Unknown incident {name}")


def _write_partition(root: Path, bd: date, day: dict[str, pd.DataFrame], manifest: dict) -> Path:
    p = ref.partition_dir(root, bd)
    p.mkdir(parents=True, exist_ok=True)
    for feed, df in day.items():
        path = p / ref.FEED_FILES[feed]
        if path.suffix == ".json":
            df.to_json(path, orient="records", lines=True)
        else:
            df.to_csv(path, index=False)
    (p / ref.MANIFEST_FILE).write_text(json.dumps(manifest, indent=2))
    return p


def _schedule(n_days: int, rng: np.random.Generator) -> list[str | None]:
    order: list[str | None] = [str(x) for x in rng.permutation(FEED_INCIDENTS)] + [None]
    return [None] + [order[(i - 1) % len(order)] for i in range(1, n_days)]


def generate(cfg: GeneratorConfig) -> dict:
    rng = np.random.default_rng(cfg.seed)
    out = Path(cfg.out_dir)
    raw_root, labels_dir = out / "raw", out / "labels"
    labels_dir.mkdir(parents=True, exist_ok=True)

    cps = _build_counterparties(cfg.n_counterparties, rng)
    fac = _build_facilities(cps, cfg.start_date, rng)
    col = _build_collateral(fac, rng)
    fx = dict(ref.BASE_FX_TO_CHF)
    days = business_days(cfg.start_date, cfg.n_days)
    schedule = cfg.incident_schedule or _schedule(len(days), rng)
    labels = LabelLog()
    prev: dict | None = None
    summary = []

    for i, bd in enumerate(days):
        if i > 0:
            _drift(fac, col, fx, rng)
        rc, genuine_ids = _rating_migrations(cps, bd, rng, genuine=i > 0)
        as_of = bd.isoformat()
        day = {
            "counterparties": cps.assign(as_of_date=as_of),
            "facilities": fac.assign(as_of_date=as_of),
            "collateral": col.assign(as_of_date=as_of),
            "rating_changes": rc,
            "fx_rates": pd.DataFrame({"currency": list(fx), "rate_to_chf": list(fx.values()), "as_of_date": as_of}),
        }
        if i > 0:  # day 0 stays clean: it is the reference "last successful load"
            inj = _Injector(cfg, rng, labels, bd)
            day["counterparties"] = _inject_counterparties(day["counterparties"], inj)
            day["facilities"] = _inject_facilities(day["facilities"], inj)
            day["collateral"] = _inject_collateral(day["collateral"], inj)
            day["rating_changes"] = _inject_rating_changes(day["rating_changes"], inj, genuine_ids)

        manifest = _manifest(day, bd, rng)
        incident = schedule[i] if i < len(schedule) else None
        applied = _apply_incident(incident, day, manifest, prev, bd, labels) if incident else False
        _write_partition(raw_root, bd, day, manifest)
        prev = {"frames": {k: v.copy() for k, v in day.items()}, "manifest": manifest}
        summary.append({"business_date": bd.isoformat(), "incident": incident if applied else None,
                        "rows": {k: int(len(v)) for k, v in day.items()}})
        log.info("wrote %s incident=%s", bd, incident if applied else None)

    label_df = labels.to_frame()
    label_df.to_csv(labels_dir / "error_labels.csv", index=False)
    (labels_dir / "generation_summary.json").write_text(json.dumps({"seed": cfg.seed, "days": summary}, indent=2))
    cfg.stats = {"days": len(days), "labels": int(len(label_df)), "counterparties": int(len(cps)), "facilities": int(len(fac)), "collateral": int(len(col))}
    return cfg.stats


def main(argv: list[str] | None = None) -> None:
    from exposure_pipeline.config import get_settings

    settings = get_settings()
    ap = argparse.ArgumentParser(description="Generate synthetic credit-risk feeds with labelled errors")
    ap.add_argument("--out", type=Path, default=settings.data_dir)
    ap.add_argument("--counterparties", type=int, default=5000)
    ap.add_argument("--days", type=int, default=10)
    ap.add_argument("--start-date", type=date.fromisoformat, default=date(2026, 1, 5))
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--error-scale", type=float, default=1.0, help="multiplier on record-level error rates")
    args = ap.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    stats = generate(GeneratorConfig(out_dir=args.out, n_counterparties=args.counterparties, n_days=args.days,
                                     start_date=args.start_date, seed=args.seed, error_scale=args.error_scale))
    print(json.dumps(stats, indent=2))


if __name__ == "__main__":
    main()
