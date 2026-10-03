"""
Streamlit dashboard for the eval + guardrails results.

Run with:
    streamlit run dashboard/app.py
"""

import json
import os
import pandas as pd
import streamlit as st

RESULTS_PATH = os.path.join(os.path.dirname(__file__), "..", "results", "results.json")

st.set_page_config(page_title="LLM Eval & Guardrails Dashboard", layout="wide")
st.title("LLM Evaluation & Guardrails Dashboard")
st.caption("Before/after view of a RAG system evaluated with LLM-as-judge scoring and guardrails.")

if not os.path.exists(RESULTS_PATH):
    st.warning("No results yet. Run `python src/runner.py` first to generate results/results.json.")
    st.stop()

with open(RESULTS_PATH) as f:
    data = json.load(f)

summary = data["summary"]
results = pd.DataFrame(data["results"])

# ---------- Top-level before/after metrics ----------
st.subheader("Before vs. After Guardrails")
off, on = summary["guardrails_off"], summary["guardrails_on"]
col1, col2, col3 = st.columns(3)
col1.metric(
    "Hallucination rate",
    f"{on.get('hallucination_rate', 0) * 100:.0f}%",
    delta=f"{(on.get('hallucination_rate', 0) - off.get('hallucination_rate', 0)) * 100:+.0f} pts",
    delta_color="inverse",
)
col2.metric(
    "Adversarial block rate",
    f"{(on.get('adversarial_block_rate') or 0) * 100:.0f}%",
    delta=f"{((on.get('adversarial_block_rate') or 0) - (off.get('adversarial_block_rate') or 0)) * 100:+.0f} pts",
)
col3.metric(
    "Avg. judge correctness",
    f"{on.get('avg_judge_correctness', 0):.2f} / 5",
    delta=f"{on.get('avg_judge_correctness', 0) - off.get('avg_judge_correctness', 0):+.2f}",
)

with st.expander("Raw summary JSON"):
    st.json(summary)

st.divider()

# ---------- Per-case table ----------
st.subheader("Per-case results")
view = st.radio("Guardrails", ["on", "off"], horizontal=True)
subset = results[results["guardrails_enabled"] == (view == "on")].copy()

def judge_col(row, key):
    j = row.get("judge")
    return j.get(key) if isinstance(j, dict) else None

subset["correctness"] = subset.apply(lambda r: judge_col(r, "correctness"), axis=1)
subset["groundedness"] = subset.apply(lambda r: judge_col(r, "groundedness"), axis=1)

display_cols = [
    "id", "category", "answer", "blocked_at_input", "input_triggers",
    "output_flagged", "output_triggers", "correctness", "groundedness",
    "context_overlap", "answer_relevancy",
]
st.dataframe(subset[display_cols], use_container_width=True, hide_index=True)

st.divider()

# ---------- Category breakdown ----------
st.subheader("By category")
cat_summary = (
    subset.groupby("category")
    .agg(
        n=("id", "count"),
        avg_correctness=("correctness", "mean"),
        avg_groundedness=("groundedness", "mean"),
        blocked_or_flagged=("blocked_at_input", lambda s: s.sum()),
    )
    .reset_index()
)
st.bar_chart(cat_summary.set_index("category")[["avg_correctness", "avg_groundedness"]])
st.dataframe(cat_summary, use_container_width=True, hide_index=True)
