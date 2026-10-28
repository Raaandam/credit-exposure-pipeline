"""Exposure aggregates: EAD and collateral-adjusted exposure by group, sector and rating band."""
from __future__ import annotations

import altair as alt
import pandas as pd
import streamlit as st

from _common import SERIES_1, empty_state, fmt_chf, query, style

st.set_page_config(page_title="Exposure aggregates", layout="wide")
st.title("Credit exposure")
st.caption("From mart_exposure_aggregate. EAD = drawn + CCF x undrawn (CHF); net = EAD - recognised collateral (capped at EAD).")

agg = query("SELECT * FROM mart_exposure_aggregate")
if agg.empty:
    empty_state("No exposure aggregates found.")
    st.stop()
agg["business_date"] = pd.to_datetime(agg["business_date"]).dt.date
dates = sorted(agg["business_date"].unique())

c1, c2, c3 = st.columns([2, 2, 2])
bd = c1.selectbox("Business date", dates[::-1])
dimension = c2.selectbox("Dimension", ["COUNTERPARTY_GROUP", "SECTOR", "RATING_BAND"])
measure = c3.radio("Measure", ["net_exposure_chf", "ead_chf"], horizontal=True,
                   format_func=lambda m: "Net (collateral-adjusted)" if m == "net_exposure_chf" else "EAD")

day = agg[(agg["business_date"] == bd) & (agg["dimension"] == dimension)]
total = agg[(agg["business_date"] == bd) & (agg["dimension"] == "COUNTERPARTY_GROUP")]
k1, k2, k3, k4 = st.columns(4)
k1.metric("Total EAD", fmt_chf(total["ead_chf"].sum()))
k2.metric("Recognised collateral", fmt_chf(total["collateral_chf"].sum()))
k3.metric("Net exposure", fmt_chf(total["net_exposure_chf"].sum()))
unmapped = total.loc[total["dimension_value"] == "UNMAPPED", "ead_chf"].sum()
k4.metric("EAD in UNMAPPED bucket", fmt_chf(unmapped), help="Facilities whose counterparty is unknown on this date (see FAC_REF_001)")

top_n = 20 if dimension == "COUNTERPARTY_GROUP" else len(day)
view = day.sort_values(measure, ascending=False).head(top_n)
st.subheader(f"{'Top 20 groups' if dimension == 'COUNTERPARTY_GROUP' else dimension.title().replace('_', ' ')} by {measure}")
chart = alt.Chart(view).mark_bar(color=SERIES_1, cornerRadiusEnd=4, size=14).encode(
    x=alt.X(f"{measure}:Q", title="CHF", axis=alt.Axis(format="~s")),
    y=alt.Y("dimension_value:N", sort="-x", title=None),
    tooltip=["dimension_value", alt.Tooltip("ead_chf:Q", format=",.0f"), alt.Tooltip("collateral_chf:Q", format=",.0f"),
             alt.Tooltip("net_exposure_chf:Q", format=",.0f"), "facility_count", "counterparty_count"],
).properties(height=max(200, 22 * len(view)))
st.altair_chart(style(chart), use_container_width=True)

st.subheader("Total net exposure over time")
trend = agg[agg["dimension"] == "COUNTERPARTY_GROUP"].groupby("business_date")[["ead_chf", "net_exposure_chf"]].sum().reset_index()
trend["business_date"] = pd.to_datetime(trend["business_date"])
line = alt.Chart(trend).mark_line(color=SERIES_1, strokeWidth=2, point=alt.OverlayMarkDef(size=64, filled=True)).encode(
    x=alt.X("business_date:T", title=None),
    y=alt.Y("net_exposure_chf:Q", title="Net exposure (CHF)", axis=alt.Axis(format="~s")),
    tooltip=[alt.Tooltip("business_date:T"), alt.Tooltip("net_exposure_chf:Q", format=",.0f"), alt.Tooltip("ead_chf:Q", format=",.0f")],
).properties(height=240)
st.altair_chart(style(line), use_container_width=True)
st.caption("Day-to-day jumps usually coincide with DQ breaks (e.g. unit changes or truncated feeds) - check the DQ page for the same date.")

with st.expander("Table view"):
    st.dataframe(day.sort_values(measure, ascending=False), hide_index=True, use_container_width=True)
