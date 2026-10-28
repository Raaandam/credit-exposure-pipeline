"""Human-in-the-loop review of LLM triage suggestions.

Nothing reaches dq_results until a named reviewer approves, edits or rejects a suggestion
here (AuditLog.record_review). The reviewer sees the evidence and the exact masked prompt
that was sent, so they can judge the suggestion on the same information the model had.
"""
from __future__ import annotations

import json

import pandas as pd
import streamlit as st

from _common import clear_cache, empty_state, engine, query

from exposure_pipeline.config import get_settings
from exposure_pipeline.triage.audit import AuditLog, ReviewError
from exposure_pipeline.triage.schemas import CATEGORY_DEFINITIONS, NEEDS_MANUAL_REVIEW

st.set_page_config(page_title="Triage review", layout="wide")
st.title("Triage review")
st.caption("LLM suggestions are proposals. Approve, edit or reject each one; only approved categories and notes are written back.")

audit = AuditLog(engine())
settings = get_settings()

with st.sidebar:
    reviewer = st.text_input("Reviewer (required)", key="reviewer")
    st.caption(f"Model: {settings.openai_model} | confidence threshold {settings.triage_confidence_threshold:.2f}")
    untriaged = query("SELECT COUNT(*) AS n FROM dq_results WHERE status IN ('FAIL','ERROR') AND triage_status = 'untriaged'")
    n_untriaged = int(untriaged["n"].iloc[0]) if not untriaged.empty else 0
    st.write(f"Breaks without a decision: **{n_untriaged}**")
    if st.button("Generate suggestions (calls OpenAI)", disabled=not settings.openai_model or n_untriaged == 0,
                 help="Requires OPENAI_API_KEY. Masked context only; results land here for review."):
        from exposure_pipeline.triage.runner import run_triage

        with st.spinner("Requesting suggestions..."):
            try:
                n = run_triage(limit=25, engine=engine())
                st.success(f"{n} suggestions recorded")
            except Exception as exc:  # noqa: BLE001
                st.error(f"Triage run failed: {type(exc).__name__}: {exc}")
        clear_cache()

pending = audit.pending()
if pending.empty:
    empty_state("No suggestions awaiting review.")
    st.stop()

pending["label"] = (pending["business_date"].astype(str) + "  " + pending["rule_id"] + "  ->  "
                    + pending["suggested_category"] + pending["confidence"].map(lambda c: "" if pd.isna(c) else f" ({c:.2f})"))
st.write(f"**{len(pending)}** suggestions awaiting review")
choice = st.selectbox("Suggestion", pending["audit_id"], format_func=lambda a: pending.set_index("audit_id").loc[a, "label"])
s = pending.set_index("audit_id").loc[choice]
brk = query("SELECT * FROM dq_results WHERE result_id = :rid", rid=s["result_id"])

left, right = st.columns([3, 2])
with left:
    st.subheader("Evidence")
    if brk.empty:
        st.warning("The break no longer exists (DQ was re-run). Reject this suggestion.")
    else:
        b = brk.iloc[0]
        st.write(f"**{b['rule_id']} - {b['rule_name']}**  \nFeed `{b['feed']}`, dimension `{b['dimension']}`, severity `{b['severity']}`")
        st.write(f"Observed: `{b['observed_value']}`  \nExpected: `{b['expected_value']}`")
        recs = json.loads(b["sample_records"] or "[]")
        if recs:
            st.dataframe(pd.DataFrame(recs), hide_index=True, use_container_width=True)
    with st.expander("Exact prompt sent to the model (masked)"):
        st.code(s["prompt_text"] or "(no prompt sent)", language="markdown")
    with st.expander("Raw model output"):
        st.code(s["output_text"] or "(none)", language="json")
    st.caption(f"model_version={s['model_version']} | prompt_hash={str(s['prompt_hash'])[:16]}... | "
               f"latency={s['latency_ms']} ms | tokens in/out={s['input_tokens']}/{s['output_tokens']}")

with right:
    st.subheader("Suggestion")
    if s["suggested_category"] == NEEDS_MANUAL_REVIEW:
        st.warning(f"Needs manual review - {s['fallback_reason']}" + (f" (model guessed `{s['llm_category']}`)" if s["llm_category"] else ""))
    else:
        st.info(f"Category: **{s['suggested_category']}** (confidence {s['confidence']:.2f})")
    categories = list(CATEGORY_DEFINITIONS)
    default = s["suggested_category"] if s["suggested_category"] in categories else (s["llm_category"] if s["llm_category"] in categories else categories[0])
    with st.form("review"):
        cat = st.selectbox("Final category", categories, index=categories.index(default),
                           help="\n".join(f"{k}: {v}" for k, v in CATEGORY_DEFINITIONS.items()))
        note = st.text_area("Root-cause note", value=s["root_cause_note"] or "", height=150)
        st.text_area("Suggested fix (model)", value=s["suggested_fix"] or "", height=80, disabled=True)
        a, e, r = st.columns(3)
        approve = a.form_submit_button("Approve", disabled=s["suggested_category"] == NEEDS_MANUAL_REVIEW)
        edit = e.form_submit_button("Save with edits")
        reject = r.form_submit_button("Reject")

    if approve or edit or reject:
        if not reviewer.strip():
            st.error("Enter your name in the sidebar first - every decision is attributed.")
        else:
            try:
                if approve:
                    audit.record_review(choice, reviewer, "approved")
                elif edit:
                    audit.record_review(choice, reviewer, "edited", final_category=cat, final_note=note)
                else:
                    audit.record_review(choice, reviewer, "rejected", final_category=cat, final_note=note)
                clear_cache()
                st.success("Decision recorded")
                st.rerun()
            except ReviewError as exc:
                st.error(str(exc))

with st.expander("Recent decisions"):
    done = query("""SELECT reviewed_at, reviewer, rule_id, business_date, suggested_category, final_category, reviewer_decision
                    FROM triage_audit WHERE reviewer_decision IS NOT NULL ORDER BY reviewed_at DESC LIMIT 50""")
    st.dataframe(done, hide_index=True, use_container_width=True)
