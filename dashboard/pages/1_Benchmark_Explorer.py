"""One simulated deployment, month by month."""

import streamlit as st
from common import COUNTER, cusum_chart, example, footnote, oe_chart, page_setup, summary

from countersignal.sequential import control_limit

page_setup("Benchmark Explorer")
st.title("Benchmark explorer")

s = summary()
if s is None:
    st.info("Benchmark results are not in this build yet.")
    st.stop()

options = {k.split("@")[0]: v["label"] for k, v in s["scenarios"].items()
           if float(k.split("@")[1]) == s["simulation"]["main_effect"]}
scenario = st.radio("Scenario", list(options), format_func=options.get, horizontal=False)
table = example(scenario)
if table is None:
    st.warning("No example run saved for this scenario.")
    st.stop()

st.markdown("The first simulated deployment of each scenario, shown as run rather than chosen "
            "for its result. Each month draws new "
            "patients from the held out pool.")
st.subheader("Is the model still calibrated?")
st.plotly_chart(oe_chart(table, tuple(s["simulation"]["oe_band"])), width="stretch")
footnote("Observed over expected readmissions for the outcome the model predicts, readmission "
         "without the intervention. The dotted line is the truth, known only because this is a "
         "benchmark. The grey line counts every patient's observed outcome, so prevented "
         "readmissions make the model look like it over-predicts.")

st.subheader("Cumulative alarm")
st.plotly_chart(cusum_chart(table, control_limit(12, s["simulation"]["alpha"])), width="stretch")
footnote("Each month adds evidence; the alarm fires when a line crosses the limit. The limit is "
         "set by simulation so that a calibrated model triggers any alarm in 12 months only 5% "
         "of the time.")

st.subheader("Monthly verdicts")
view = table[["window", "n", "n_treated", "true_oe", "oe_naive", "naive_oe_verdict",
              "oe_ipw", "oe_ipw_lo", "oe_ipw_hi", "oe_verdict", "auc_ipw", "true_auc", "ess"]]
st.dataframe(view.rename(columns={
    "window": "Month", "n": "Patients", "n_treated": "Treated", "true_oe": "True O/E",
    "oe_naive": "Standard O/E", "naive_oe_verdict": "Standard verdict",
    "oe_ipw": "CounterSignal O/E", "oe_ipw_lo": "Lower 95%", "oe_ipw_hi": "Upper 95%",
    "oe_verdict": "CounterSignal verdict", "auc_ipw": "CounterSignal AUC", "true_auc": "True AUC",
    "ess": "Effective untreated sample"}).round(3), hide_index=True, width="stretch")
st.markdown(f"<span style='color:{COUNTER}'>■</span> Verdicts use an acceptable O/E range of "
            f"{s['simulation']['oe_band'][0]} to {s['simulation']['oe_band'][1]}: **stable** when "
            "the whole interval is inside it, **drift** when the whole interval is outside it, and "
            "**not enough evidence yet** otherwise.", unsafe_allow_html=True)
