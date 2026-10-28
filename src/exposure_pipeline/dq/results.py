"""Persist DQ results to the ``dq_results`` table."""
from __future__ import annotations

from datetime import date

import pandas as pd
from sqlalchemy.engine import Engine

from exposure_pipeline.dq.engine import DQResult
from exposure_pipeline.storage.db import dumps, replace_partition
from exposure_pipeline.storage.tables import dq_results


def results_frame(results: list[DQResult]) -> pd.DataFrame:
    rows = []
    for r in results:
        d = r.to_dict()
        d["failed_keys"] = dumps(d["failed_keys"])
        d["sample_records"] = dumps(d["sample_records"])
        d["triage_status"] = "untriaged" if r.is_break else "n/a"
        rows.append(d)
    return pd.DataFrame(rows)


def write_results(engine: Engine, results: list[DQResult], business_date: date) -> int:
    """Replace the day's results. A rerun is a fresh evaluation, so triage status resets to
    'untriaged'; earlier suggestions/decisions remain in triage_audit (keyed by result_id)."""
    with engine.begin() as conn:
        return replace_partition(conn, dq_results, results_frame(results), business_date)
