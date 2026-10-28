"""Stopwatch for measuring manual vs automated reconciliation effort.

Any reported reduction in reconciliation effort should come from measurements, not
estimates. This script records analyst wall-clock time for one daily run in each mode
and summarises it.

    # interactive: shows the checklist, press Enter to start / stop
    python scripts/time_reconciliation.py record --mode manual    --business-date 2026-01-07 --analyst MS
    python scripts/time_reconciliation.py record --mode automated --business-date 2026-01-08 --analyst MS
    # log a time measured elsewhere
    python scripts/time_reconciliation.py record --mode manual --business-date 2026-01-09 --analyst MS --minutes 41
    # summary (+ write it into the README placeholder)
    python scripts/time_reconciliation.py summary --update-readme

Protocol to keep the comparison fair (printed by ``protocol``):
* Same analyst(s), comparable days (similar break counts), at least 5 runs per mode.
* Alternate the order (manual first on some days, automated first on others) and never
  time both modes on the same business date - the second pass would benefit from memory.
* "Done" means the same thing in both modes: every break found, explained and either
  resolved or escalated. Record breaks found so effort is compared at equal coverage.
* Automated time = human time only (dashboard review + triage approvals); machine run time
  is reported separately by the pipeline logs.
"""
from __future__ import annotations

import argparse
import csv
import sys
import time
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

TIMINGS = ROOT / "eval" / "reconciliation_timings.csv"
FIELDS = ["recorded_at", "mode", "business_date", "analyst", "minutes", "breaks_found", "notes"]

CHECKLISTS = {
    "manual": """MANUAL reconciliation (legacy process)
  1. Run legacy/onprem_batch.py for the date and keep its printed control report.
  2. Open manifest.json; compare record counts and control totals with the printed counts
     and with a spreadsheet SUM of drawn/limit/market value.
  3. Spreadsheet checks: missing ratings/sectors/currencies (filters), duplicate ids
     (COUNTIF), facility->counterparty and collateral->facility lookups (VLOOKUP/XLOOKUP).
  4. Compare exposure by group/sector/rating band with the previous day's report; explain moves.
  5. Write a short note per break (cause + owner).""",
    "automated": """AUTOMATED reconciliation (this project)
  1. Open the dashboard DQ page for the date (pipeline + DQ already ran).
  2. Review every FAIL/ERROR row: sample records, observed vs expected.
  3. On the Triage Review page approve / edit / reject each suggestion.
  4. Check the exposure page for unexplained moves.
  5. Note anything escalated.""",
}


def _append(row: dict) -> None:
    TIMINGS.parent.mkdir(parents=True, exist_ok=True)
    new = not TIMINGS.exists()
    with TIMINGS.open("a", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=FIELDS)
        if new:
            w.writeheader()
        w.writerow(row)


def record(args: argparse.Namespace) -> None:
    if args.minutes is None:
        print(CHECKLISTS[args.mode])
        input("\nPress Enter to START the timer...")
        t0 = time.perf_counter()
        input("Press Enter when the reconciliation is DONE...")
        minutes = (time.perf_counter() - t0) / 60
    else:
        minutes = args.minutes
    breaks = args.breaks_found if args.breaks_found is not None else input("Breaks found: ").strip() or ""
    row = {"recorded_at": datetime.now().isoformat(timespec="seconds"), "mode": args.mode, "business_date": args.business_date,
           "analyst": args.analyst, "minutes": round(minutes, 2), "breaks_found": breaks, "notes": args.notes or ""}
    _append(row)
    print(f"recorded {row['minutes']} min ({args.mode}) -> {TIMINGS}")


def summary(args: argparse.Namespace) -> None:
    import pandas as pd

    if not TIMINGS.exists():
        print("no timings recorded yet")
        return
    df = pd.read_csv(TIMINGS)
    g = df.groupby("mode")["minutes"].agg(["count", "mean", "median", "std"])
    print(g.to_string())
    if not {"manual", "automated"} <= set(g.index):
        print("need both modes for a comparison")
        return
    m, a = g.loc["manual"], g.loc["automated"]
    reduction = 1 - a["mean"] / m["mean"]
    small = min(m["count"], a["count"]) < 5
    md = "\n".join([
        f"_Measured with `scripts/time_reconciliation.py`; last updated {datetime.now():%Y-%m-%d}._",
        "",
        "| Mode | Runs | Mean minutes / daily run | Median |",
        "|---|---|---|---|",
        f"| Manual (legacy) | {int(m['count'])} | {m['mean']:.1f} | {m['median']:.1f} |",
        f"| Automated (DQ engine + triage review) | {int(a['count'])} | {a['mean']:.1f} | {a['median']:.1f} |",
        "",
        f"Reduction in analyst time per daily run: **{reduction:.0%}**"
        + (" (fewer than 5 runs per mode - treat as indicative)" if small else ""),
    ])
    print("\n" + md)
    if args.update_readme:
        from exposure_pipeline.reporting import update_readme_section

        print("README updated" if update_readme_section("reconciliation_timing", md) else "README marker not found")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("record")
    r.add_argument("--mode", choices=["manual", "automated"], required=True)
    r.add_argument("--business-date", required=True)
    r.add_argument("--analyst", required=True)
    r.add_argument("--minutes", type=float, help="log a duration instead of using the stopwatch")
    r.add_argument("--breaks-found", type=int)
    r.add_argument("--notes")
    s = sub.add_parser("summary")
    s.add_argument("--update-readme", action="store_true")
    sub.add_parser("protocol")
    args = ap.parse_args()
    if args.cmd == "record":
        record(args)
    elif args.cmd == "summary":
        summary(args)
    else:
        print(__doc__[__doc__.index("Protocol"):])


if __name__ == "__main__":
    main()
