"""Data-quality breaks dashboard (home page).

    streamlit run dashboard/app.py
"""
from __future__ import annotations

import json

import altair as alt
import pandas as pd
import streamlit as st

from _common import SEQUENTIAL, SERIES_1, empty_state, query, style

st.set_page_config(page_title="Credit Exposure DQ", layout="wide")
st.title("Data-quality breaks")
st.caption("Rule results from dq_results. A break is a rule with status FAIL or ERROR for a business date.")

results = query("SELECT * FROM dq_results")
if results.empty:
    empty_state("No DQ results found.")
    st.stop()
results["business_date"] = pd.to_datetime(results["business_date"]).dt.date

# ---- filters (one row, above the charts)
dates = sorted(results["business_date"].unique())
c1, c2, c3, c4 = st.columns([2, 2, 2, 2])
date_range = c1.select_slider("Business dates", options=dates, value=(dates[0], dates[-1]))
feeds = c2.multiselect("Feed", sorted(results["feed"].dropna().unique()))
dims = c3.multiselect("Dimension", sorted(results["dimension"].dropna().unique()))
sev = c4.multiselect("Severity", ["critical", "high", "medium", "low"])

f = results[(results["business_date"] >= date_range[0]) & (results["business_date"] <= date_range[1])]
if feeds:
    f = f[f["feed"].isin(feeds)]
if dims:
    f = f[f["dimension"].isin(dims)]
if sev:
    f = f[f["severity"].isin(sev)]
breaks = f[f["status"].isin(["FAIL", "ERROR"])]

# ---- headline numbers
k1, k2, k3, k4, k5 = st.columns(5)
k1.metric("Rule evaluations", f"{len(f):,}")
k2.metric("Breaks", f"{len(breaks):,}")
k3.metric("Failing records", f"{int(breaks['failed_count'].fillna(0).sum()):,}")
k4.metric("Rules with breaks", f"{breaks['rule_id'].nunique():,}")
k5.metric("Awaiting triage", f"{int((breaks['triage_status'] == 'untriaged').sum()):,}")

if breaks.empty:
    st.success("No breaks for the selected filters.")
    st.stop()

# ---- breaks by feed x day (heatmap) and by rule (bar)
left, right = st.columns(2)
with left:
    st.subheader("Breaks by feed and day")
    heat = breaks.groupby(["business_date", "feed"]).size().reset_index(name="breaks")
    heat["business_date"] = heat["business_date"].astype(str)
    chart = alt.Chart(heat).mark_rect(cornerRadius=2, stroke="white", strokeWidth=2).encode(
        x=alt.X("business_date:O", title="Business date"),
        y=alt.Y("feed:N", title=None),
        color=alt.Color("breaks:Q", scale=alt.Scale(range=SEQUENTIAL), title="Breaks"),
        tooltip=["business_date", "feed", "breaks"],
    ).properties(height=260)
    st.altair_chart(style(chart), use_container_width=True)
with right:
    st.subheader("Breaks by rule")
    by_rule = breaks.groupby(["rule_id", "rule_name"]).size().reset_index(name="days_breaking").sort_values("days_breaking", ascending=False).head(15)
    chart = alt.Chart(by_rule).mark_bar(color=SERIES_1, cornerRadiusEnd=4, size=14).encode(
        x=alt.X("days_breaking:Q", title="Days with a break"),
        y=alt.Y("rule_id:N", sort="-x", title=None),
        tooltip=["rule_id", "rule_name", "days_breaking"],
    ).properties(height=260)
    st.altair_chart(style(chart), use_container_width=True)

st.subheader("Breaks by dimension")
by_dim = breaks.groupby("dimension").agg(breaks=("rule_id", "size"), failing_records=("failed_count", "sum")).reset_index()
st.dataframe(by_dim, hide_index=True, use_container_width=True)

# ---- detail table
st.subheader("Break detail")
cols = ["business_date", "status", "severity", "rule_id", "rule_name", "feed", "dimension", "failed_count",
        "total_count", "observed_value", "expected_value", "triage_status", "triage_category"]
detail = breaks.sort_values(["business_date", "severity", "rule_id"], ascending=[False, True, True])
st.dataframe(detail[cols], hide_index=True, use_container_width=True)

choice = st.selectbox("Inspect a break", detail["result_id"].tolist())
if choice:
    row = detail.set_index("result_id").loc[choice]
    st.write(f"**{row['rule_id']} - {row['rule_name']}** ({row['status']}, {row['severity']})")
    st.write(f"Observed: `{row['observed_value']}`  |  Expected: `{row['expected_value']}`  |  {row['message']}")
    recs = json.loads(row["sample_records"] or "[]")
    if recs:
        st.caption("Sample failing records (internal view, unmasked)")
        st.dataframe(pd.DataFrame(recs), hide_index=True, use_container_width=True)
    keys = json.loads(row["failed_keys"] or "[]")
    if keys:
        st.caption(f"Failing keys (first {len(keys)}): " + ", ".join(keys))
