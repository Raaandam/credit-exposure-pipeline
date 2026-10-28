"""Migration parity check: legacy batch output vs new pipeline mart for one business date.

Run the legacy batch and the new pipeline on the same partition, then:

    python scripts/parity_check.py --business-date 2026-01-05

Compares EAD / net exposure per (dimension, dimension_value) and exits non-zero on any
difference above the tolerance. Use a clean day (day 0 of the generator) for a strict check.
On days with near-duplicate facilities small differences are expected: the legacy SQL uses
``DISTINCT ON`` without ORDER BY (arbitrary survivor), the new pipeline keeps the first
occurrence in file order - one of the defects the migration fixed deliberately.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from exposure_pipeline.storage.db import get_engine, read_sql  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--business-date", required=True)
    ap.add_argument("--tolerance", type=float, default=0.01, help="absolute CHF tolerance per row")
    args = ap.parse_args()
    engine = get_engine()
    legacy = read_sql(engine, "SELECT dimension, dimension_value, ead_chf, net_exposure_chf FROM legacy_exposure_report")
    new = read_sql(engine, """SELECT dimension, dimension_value, ead_chf, net_exposure_chf FROM mart_exposure_aggregate
                              WHERE business_date = :bd""", bd=args.business_date)
    m = legacy.merge(new, on=["dimension", "dimension_value"], how="outer", suffixes=("_legacy", "_new"), indicator=True)
    m["ead_diff"] = (m["ead_chf_new"].fillna(0) - m["ead_chf_legacy"].fillna(0)).abs()
    m["net_diff"] = (m["net_exposure_chf_new"].fillna(0) - m["net_exposure_chf_legacy"].fillna(0)).abs()
    bad = m[(m["_merge"] != "both") | (m["ead_diff"] > args.tolerance) | (m["net_diff"] > args.tolerance)]
    print(f"rows compared: {len(m)}, mismatches: {len(bad)}")
    if len(bad):
        print(bad.head(20).to_string())
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
