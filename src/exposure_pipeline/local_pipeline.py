"""Laptop-scale pipeline: pandas curation + DQ engine, writing to SQLite or Postgres.

Same stages and tables as the Spark/Airflow pipeline, without docker. Useful for demos,
for developing DQ rules, and for producing data the dashboard and triage can run on.

    python -m exposure_pipeline.local_pipeline --all
    python -m exposure_pipeline.local_pipeline --business-date 2026-01-07
"""
from __future__ import annotations

import argparse
import json
import logging
import time
import uuid
from datetime import date, datetime, timezone
from pathlib import Path

import pandas as pd
from sqlalchemy.engine import Engine

from exposure_pipeline import reference as ref
from exposure_pipeline.config import get_settings
from exposure_pipeline.dq.backends.pandas_backend import PandasBackend
from exposure_pipeline.dq.engine import DQEngine
from exposure_pipeline.dq.results import write_results
from exposure_pipeline.dq.rules import load_rules
from exposure_pipeline.storage.db import ensure_schema, get_engine, replace_partition
from exposure_pipeline.storage.tables import TABLES_BY_DATASET, load_audit
from exposure_pipeline.transform.pandas_impl import curate_all

log = logging.getLogger(__name__)


def read_raw_partition(raw_root: Path, business_date: date) -> tuple[dict[str, pd.DataFrame], dict]:
    """Read one partition as strings (schema-on-read). Missing files are simply absent."""
    part = ref.partition_dir(raw_root, business_date)
    datasets: dict[str, pd.DataFrame] = {}
    for feed, fname in ref.FEED_FILES.items():
        path = part / fname
        if not path.exists():
            log.warning("%s: feed %s not found at %s", business_date, feed, path)
            continue
        if path.suffix == ".json":
            datasets[f"raw.{feed}"] = pd.read_json(path, lines=True, dtype=False, convert_dates=False)
        else:
            datasets[f"raw.{feed}"] = pd.read_csv(path, dtype=str)
    manifest_path = part / ref.MANIFEST_FILE
    manifest = json.loads(manifest_path.read_text())["feeds"] if manifest_path.exists() else {}
    return datasets, manifest


def _load_audit_frame(bd: date, run_id: str, raw: dict, manifest: dict, raw_root: Path) -> pd.DataFrame:
    now = datetime.now(timezone.utc).replace(tzinfo=None)
    rows = []
    for feed, fname in ref.FEED_FILES.items():
        m = manifest.get(feed, {})
        df = raw.get(f"raw.{feed}")
        rows.append({
            "business_date": bd, "feed": feed, "run_id": run_id,
            "status": "SUCCESS" if df is not None else "MISSING",
            "row_count": None if df is None else len(df),
            "source_path": str(ref.partition_dir(raw_root, bd) / fname),
            "source_system": m.get("source_system"), "manifest_business_date": m.get("business_date"),
            "manifest_record_count": m.get("record_count"), "extract_timestamp": m.get("extract_timestamp"),
            "loaded_at": now,
        })
    return pd.DataFrame(rows)


def run_business_date(business_date: date, engine: Engine, raw_root: Path, dq: DQEngine) -> dict:
    t0 = time.perf_counter()
    run_id = f"local-{business_date:%Y%m%d}-{uuid.uuid4().hex[:8]}"
    raw, manifest = read_raw_partition(raw_root, business_date)

    curated: dict[str, pd.DataFrame] = {}
    if "raw.counterparties" in raw and "raw.facilities" in raw:
        curated = curate_all(raw, business_date)
    else:
        log.error("%s: core feed missing, curated layer not refreshed", business_date)

    with engine.begin() as conn:  # one transaction per day: all-or-nothing, rerunnable
        for name, table in TABLES_BY_DATASET.items():
            if name in curated:
                replace_partition(conn, table, curated[name], business_date)
        replace_partition(conn, load_audit, _load_audit_frame(business_date, run_id, raw, manifest, raw_root), business_date)
    t_load = time.perf_counter() - t0

    results = dq.run({**raw, **curated, "manifest": manifest}, business_date, run_id=run_id)
    write_results(engine, results, business_date)
    t_total = time.perf_counter() - t0
    summary = {
        "business_date": business_date.isoformat(), "run_id": run_id,
        "rules": len(results), "breaks": sum(r.is_break for r in results),
        "load_seconds": round(t_load, 2), "total_seconds": round(t_total, 2),
    }
    log.info("%s", summary)
    return summary


def available_dates(raw_root: Path) -> list[date]:
    return sorted(date.fromisoformat(p.name.split("=", 1)[1]) for p in raw_root.glob("business_date=*") if p.is_dir())


def main(argv: list[str] | None = None) -> None:
    settings = get_settings()
    ap = argparse.ArgumentParser(description="Run curation + DQ locally with pandas")
    ap.add_argument("--business-date", type=date.fromisoformat)
    ap.add_argument("--all", action="store_true", help="process every partition found")
    ap.add_argument("--raw-dir", type=Path, default=settings.raw_dir)
    ap.add_argument("--database-url", default=settings.database_url)
    args = ap.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s %(message)s")

    engine = get_engine(args.database_url)
    ensure_schema(engine)
    dq = DQEngine(load_rules(), PandasBackend())
    dates = available_dates(args.raw_dir) if args.all else [args.business_date]
    if not dates or dates == [None]:
        ap.error("pass --business-date or --all (and generate data first)")
    summaries = [run_business_date(d, engine, args.raw_dir, dq) for d in dates]
    print(json.dumps(summaries, indent=2))


if __name__ == "__main__":
    main()
