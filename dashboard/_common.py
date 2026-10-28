"""Shared helpers for the Streamlit pages (DB access, caching, chart styling)."""
from __future__ import annotations

import sys
from pathlib import Path

import altair as alt
import pandas as pd
import streamlit as st

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "src") not in sys.path:  # allow `streamlit run dashboard/app.py` without pip install -e .
    sys.path.insert(0, str(ROOT / "src"))

from exposure_pipeline.storage.db import ensure_schema, get_engine, read_sql  # noqa: E402

# Chart styling: one sequential blue ramp for magnitude, single-series bars in slot-1 blue,
# recessive axes/grid. Status (FAIL/ERROR) is always shown with a text label, never colour alone.
SERIES_1 = "#2a78d6"
SEQUENTIAL = ["#cde2fb", "#86b6ef", "#3987e5", "#1c5cab", "#0d366b"]
MUTED = "#898781"
GRID = "#e1e0d9"


@st.cache_resource
def engine():
    eng = get_engine()
    ensure_schema(eng)
    return eng


@st.cache_data(ttl=60)
def query(sql: str, **params) -> pd.DataFrame:
    return read_sql(engine(), sql, **params)


def clear_cache() -> None:
    query.clear()


def style(chart: alt.Chart) -> alt.Chart:
    return (chart.configure_axis(labelColor=MUTED, titleColor=MUTED, gridColor=GRID, domainColor=GRID, tickColor=GRID)
                 .configure_view(strokeWidth=0))


def fmt_chf(x: float) -> str:
    if pd.isna(x):
        return "n/a"
    for unit, div in (("bn", 1e9), ("m", 1e6), ("k", 1e3)):
        if abs(x) >= div:
            return f"CHF {x / div:,.2f}{unit}"
    return f"CHF {x:,.0f}"


def empty_state(msg: str = "No data yet.") -> None:
    st.info(f"{msg} Generate data and run the pipeline first:\n\n"
            "```\npython -m exposure_pipeline.generator.generate_feeds\n"
            "python -m exposure_pipeline.local_pipeline --all\n```")
