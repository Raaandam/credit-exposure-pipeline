""""On-prem" baseline: the legacy nightly batch this project migrates away from.

Deliberately written the way such jobs typically look - plain Python + psycopg2 against a
single Postgres server, reading files from a shared drive:

* full reload every night (TRUNCATE + COPY), no business-date partitioning, so history is
  lost and a rerun of an older date overwrites today's data;
* aggregation in one big SQL statement (GROUPING SETS);
* the only "data quality" control is printed row counts that an analyst compares by hand
  against the source control file and the previous day's report;
* single-threaded, no retries, no orchestration (cron), credentials from env vars.

It is kept runnable as the functional baseline: its aggregates should equal the new
pipeline's ``mart_exposure_aggregate`` for the same clean input, which is how the migration
is regression-tested (see README "Migration parity check").

    python legacy/onprem_batch.py --input-dir data/raw/business_date=2026-01-05 \
        --dsn "host=localhost dbname=credit user=credit password=credit"
"""
from __future__ import annotations

import argparse
import csv
import io
import json
import os
import sys
import time
from pathlib import Path

import psycopg2  # legacy job talks to the DB driver directly

STAGING_DDL = """
DROP TABLE IF EXISTS legacy_stg_counterparty, legacy_stg_facility, legacy_stg_collateral, legacy_stg_fx;
CREATE TABLE legacy_stg_counterparty (counterparty_id TEXT, counterparty_name TEXT, lei TEXT, group_id TEXT,
    ultimate_parent_id TEXT, country TEXT, sector TEXT, internal_rating TEXT, as_of_date TEXT);
CREATE TABLE legacy_stg_facility (facility_id TEXT, counterparty_id TEXT, facility_type TEXT, currency TEXT,
    limit_amount TEXT, drawn_amount TEXT, maturity_date TEXT, source_system TEXT, as_of_date TEXT);
CREATE TABLE legacy_stg_collateral (collateral_id TEXT, facility_id TEXT, collateral_type TEXT, currency TEXT,
    market_value TEXT, haircut TEXT, as_of_date TEXT);
CREATE TABLE legacy_stg_fx (currency TEXT, rate_to_chf TEXT, as_of_date TEXT);
CREATE TABLE IF NOT EXISTS legacy_exposure_report (dimension TEXT, dimension_value TEXT, ead_chf DOUBLE PRECISION,
    collateral_chf DOUBLE PRECISION, net_exposure_chf DOUBLE PRECISION, facility_count INTEGER, run_ts TIMESTAMP);
"""

# Same business logic as the new pipeline (EAD = drawn + CCF*undrawn; collateral capped at EAD),
# expressed as the kind of monolithic SQL the legacy job used.
AGGREGATE_SQL = """
TRUNCATE legacy_exposure_report;
WITH cp AS (
    SELECT DISTINCT ON (counterparty_id) counterparty_id, group_id, sector, internal_rating
    FROM legacy_stg_counterparty
), fac AS (
    SELECT DISTINCT ON (facility_id) facility_id, counterparty_id, UPPER(TRIM(facility_type)) AS facility_type,
           UPPER(TRIM(currency)) AS currency, NULLIF(limit_amount, '')::float AS limit_amount,
           NULLIF(drawn_amount, '')::float AS drawn_amount
    FROM legacy_stg_facility
), ccf(facility_type, ccf) AS (
    VALUES ('TERM_LOAN', 1.0), ('REVOLVER', 0.4), ('TRADE_FINANCE', 0.2), ('GUARANTEE', 0.5), ('OVERDRAFT', 0.1)
), fx AS (
    SELECT UPPER(TRIM(currency)) AS currency, rate_to_chf::float AS rate FROM legacy_stg_fx
), coll AS (
    SELECT c.facility_id,
           SUM(NULLIF(c.market_value, '')::float * (1 - LEAST(GREATEST(NULLIF(c.haircut, '')::float, 0), 1)) * fx.rate) AS value_chf
    FROM (SELECT DISTINCT ON (collateral_id) * FROM legacy_stg_collateral) c
    JOIN fx ON fx.currency = UPPER(TRIM(c.currency))
    GROUP BY c.facility_id
), exposure AS (
    SELECT f.facility_id,
           COALESCE(cp.group_id, 'UNMAPPED') AS group_id,
           CASE WHEN cp.counterparty_id IS NULL THEN 'UNMAPPED'
                WHEN UPPER(TRIM(cp.sector)) IN ('BANKS','INSURANCE','ASSET_MANAGEMENT','SOVEREIGN','ENERGY','REAL_ESTATE',
                     'INDUSTRIALS','TECHNOLOGY','HEALTHCARE','CONSUMER','UTILITIES','TRANSPORT') THEN UPPER(TRIM(cp.sector))
                ELSE 'UNMAPPED' END AS sector,
           CASE WHEN cp.counterparty_id IS NULL THEN 'UNMAPPED'
                WHEN UPPER(TRIM(cp.internal_rating)) IN ('AAA','AA','A') THEN 'IG_HIGH'
                WHEN UPPER(TRIM(cp.internal_rating)) = 'BBB' THEN 'IG_LOW'
                WHEN UPPER(TRIM(cp.internal_rating)) IN ('BB','B') THEN 'SUB_IG'
                WHEN UPPER(TRIM(cp.internal_rating)) = 'CCC' THEN 'WATCHLIST'
                WHEN UPPER(TRIM(cp.internal_rating)) = 'D' THEN 'DEFAULT'
                ELSE 'UNRATED' END AS rating_band,
           (f.drawn_amount + ccf.ccf * GREATEST(f.limit_amount - f.drawn_amount, 0)) * fx.rate AS ead_chf,
           coll.value_chf
    FROM fac f
    LEFT JOIN cp ON cp.counterparty_id = f.counterparty_id
    LEFT JOIN ccf ON ccf.facility_type = f.facility_type
    LEFT JOIN fx ON fx.currency = f.currency
    LEFT JOIN coll ON coll.facility_id = f.facility_id
), capped AS (
    SELECT *, CASE WHEN ead_chf IS NULL THEN NULL ELSE LEAST(COALESCE(value_chf, 0), ead_chf) END AS collateral_chf
    FROM exposure
)
INSERT INTO legacy_exposure_report
SELECT CASE WHEN GROUPING(group_id) = 0 THEN 'COUNTERPARTY_GROUP'
            WHEN GROUPING(sector) = 0 THEN 'SECTOR' ELSE 'RATING_BAND' END,
       COALESCE(group_id, sector, rating_band),
       SUM(ead_chf), SUM(collateral_chf), SUM(ead_chf - collateral_chf), COUNT(*), NOW()
FROM capped
GROUP BY GROUPING SETS ((group_id), (sector), (rating_band));
"""

FILES = {
    "legacy_stg_counterparty": "counterparties.csv",
    "legacy_stg_facility": "facilities.csv",
    "legacy_stg_fx": "fx_rates.csv",
}


def _copy_csv(cur, table: str, path: Path) -> int:
    with path.open(newline="", encoding="utf-8") as fh:
        cur.copy_expert(f"COPY {table} FROM STDIN WITH (FORMAT csv, HEADER true)", fh)
    return cur.rowcount


def _copy_jsonl(cur, table: str, path: Path, columns: list[str]) -> int:
    # The collateral system sends JSON lines; the legacy job flattens them to CSV in memory.
    buf = io.StringIO()
    writer = csv.writer(buf)
    n = 0
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            rec = json.loads(line)
            writer.writerow(["" if rec.get(c) is None else rec.get(c) for c in columns])
            n += 1
    buf.seek(0)
    cur.copy_expert(f"COPY {table} FROM STDIN WITH (FORMAT csv)", buf)
    return n


def main() -> int:
    ap = argparse.ArgumentParser(description="Legacy nightly exposure batch (baseline)")
    ap.add_argument("--input-dir", type=Path, required=True)
    ap.add_argument("--dsn", default=os.getenv("LEGACY_PG_DSN", "host=localhost dbname=credit user=credit password=credit"))
    args = ap.parse_args()

    t0 = time.time()
    conn = psycopg2.connect(args.dsn)
    try:
        with conn, conn.cursor() as cur:
            cur.execute(STAGING_DDL)
            counts = {t: _copy_csv(cur, t, args.input_dir / f) for t, f in FILES.items()}
            counts["legacy_stg_collateral"] = _copy_jsonl(
                cur, "legacy_stg_collateral", args.input_dir / "collateral.json",
                ["collateral_id", "facility_id", "collateral_type", "currency", "market_value", "haircut", "as_of_date"])
            cur.execute(AGGREGATE_SQL)
            cur.execute("SELECT dimension, COUNT(*), SUM(ead_chf) FROM legacy_exposure_report GROUP BY dimension")
            totals = cur.fetchall()
    finally:
        conn.close()

    # "Control report": this printout is what analysts used to tick off manually.
    print("Rows loaded:")
    for table, n in counts.items():
        print(f"  {table:<26} {n:>10,}")
    print("Report totals:")
    for dim, rows, ead in totals:
        print(f"  {dim:<20} rows={rows:>6,} ead_chf={ead:,.2f}")
    print(f"Finished in {time.time() - t0:.1f}s")
    return 0


if __name__ == "__main__":
    sys.exit(main())
