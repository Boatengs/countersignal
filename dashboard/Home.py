"""CounterSignal dashboard: home."""

import pandas as pd
import streamlit as st
from common import METHOD_LABELS, page_setup, summary

page_setup("Home")

st.title("CounterSignal")
st.markdown("#### Performance monitoring for deployed clinical risk models")
st.markdown(
    "When a deployed risk model flags a patient and the care team intervenes, the outcome the "
    "model predicted may be prevented. Standard monitoring compares predictions with recorded "
    "outcomes, so an effective intervention appears as model miscalibration and can trigger false "
    "alarms or unnecessary retraining. CounterSignal evaluates the model against the outcome it was "
    "validated to predict: it retains untreated patients, weights them by the inverse probability "
    "of remaining untreated, and reports calibration and discrimination with 95% intervals. Where "
    "the data are too limited to reach a conclusion, as is common in smaller hospitals, the verdict "
    "states that the evidence is insufficient rather than raising an alarm.")

s = summary()
if s is None:
    st.info("Benchmark results are not in this build yet. Run the Data and benchmark workflow.")
    st.stop()

st.subheader("Benchmark on real hospital records")
d, m, sim = s["data"], s["model"], s["simulation"]
st.markdown(
    f"{d['cohort']:,} first diabetic inpatient encounters from 130 U.S. hospitals (1999 to 2008, "
    f"UCI). A readmission model (AUC {m['reference_auc']:.3f} at validation) is deployed in a "
    f"simulation: alerted patients are treated and treatment prevents "
    f"{sim['main_effect']:.0%} of their readmissions. The real outcomes stand in for what would "
    f"have happened without treatment, so the true performance in every month is known. "
    f"{sim['replicates']} simulated deployments per scenario, 12 months each.")


def pct(x):
    return "n/a" if x is None else f"{x:.0%}"


rows = []
for key, entry in s["scenarios"].items():
    scenario, effect = key.split("@")
    if float(effect) != sim["main_effect"]:
        continue
    for method, label in METHOD_LABELS.items():
        e = entry[method]
        rows.append({
            "Scenario": entry["label"], "Method": label,
            "Any alarm in 12 months": pct(e["any_alarm_rate"]),
            "Alarm before the change": pct(e.get("alarm_before_shift_rate")),
            "Caught the change": pct(e.get("detected_rate")),
            "Months to alarm (median)": ("n/a" if e.get("median_delay_windows") is None
                                         else f"{e['median_delay_windows'] + 1:.0f}"),
            "Mean error in O/E": f"{e['oe_abs_error']:.3f}",
            "Interval covers truth": pct(e["oe_ci_coverage"]),
        })
st.dataframe(pd.DataFrame(rows), hide_index=True, width="stretch")
st.caption(
    "Any alarm: share of simulated deployments where the cumulative alarm fired. For scenarios "
    "where nothing changed, every alarm is false. Months to alarm counts the first month of the "
    "change as month 1. O/E is observed over expected readmissions; 1.00 means calibrated. "
    "All three methods use the same alarm rule and the same 5% planned false alarm rate.")
st.page_link("pages/1_Benchmark_Explorer.py", label="Explore a simulated deployment month by month",
             icon="📈")
st.page_link("pages/2_Run_On_Your_Data.py", label="Run CounterSignal on your own deployment log", icon="🧪")
st.page_link("pages/3_Method.py", label="Method, assumptions and limits", icon="📄")
