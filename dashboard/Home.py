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
    return "n/a" if x is None else f"{x * 100:.1f}%"


main = sim["main_effect"]
sc = {k.split("@")[0]: v for k, v in s["scenarios"].items() if float(k.split("@")[1]) == main}
cols = list(METHOD_LABELS.values())

st.markdown("**False alarms when the model has not changed** (share of 12 month deployments with "
            "any cumulative alarm; planned rate 5%)")
rows = []
for key in ("stable", "case_mix", "small_hospital"):
    if key in sc:
        rows.append({"Scenario": sc[key]["label"],
                     **{METHOD_LABELS[m]: pct(sc[key][m]["any_alarm_rate"]) for m in METHOD_LABELS}})
st.dataframe(pd.DataFrame(rows), hide_index=True, width="stretch")

if "high_risk_drift" in sc:
    d = sc["high_risk_drift"]
    st.markdown(f"**Detecting real drift:** {d['label'].lower()}")
    measures = [
        ("Deployments where the change was caught", lambda e: pct(e["detected_rate"])),
        ("False alarm before the change began", lambda e: pct(e["alarm_before_shift_rate"])),
        ("Median month of first alarm after the change",
         lambda e: "n/a" if e["median_delay_windows"] is None else f"{e['median_delay_windows'] + 1:.0f}"),
    ]
    st.dataframe(pd.DataFrame([{"Measure": name, **{METHOD_LABELS[m]: f(d[m]) for m in METHOD_LABELS}}
                               for name, f in measures]), hide_index=True, width="stretch")

st.caption(
    f"{sim['replicates']} simulated deployments per scenario. All three methods use the same "
    "cumulative alarm rule and limit. The standard monitor compares predictions with every "
    "patient's recorded outcome; the untreated only view drops treated patients without "
    "reweighting. Treatment effect 40%; results for 20% and 60% are in the case study.")

with st.expander("What the correction costs"):
    e = sc["stable"]
    st.markdown(
        f"Reweighting leaves about {e['mean_ess']:,.0f} effective patients out of 2,000 a month, so "
        f"monthly intervals are wider. With the model unchanged, CounterSignal's monthly "
        f"observed/expected was off by {e['countersignal']['oe_abs_error']:.3f} on average, "
        f"against {e['untreated_only']['oe_abs_error']:.3f} for the untreated only view, and it "
        f"returned \"not enough evidence yet\" in {e['countersignal']['insufficient_windows_mean']:.1f} "
        f"of 12 months against {e['untreated_only']['insufficient_windows_mean']:.1f}. Its 95% "
        f"intervals contained the true value in {pct(e['countersignal']['oe_ci_coverage'])} of months. "
        "Single months rarely settle the question at this volume; the cumulative alarm carries "
        "the evidence across months.")

st.page_link("pages/1_Benchmark_Explorer.py", label="Explore a simulated deployment month by month",
             icon="📈")
st.page_link("pages/2_Run_On_Your_Data.py", label="Run CounterSignal on your own deployment log", icon="🧪")
st.page_link("pages/3_Method.py", label="Method, assumptions and limits", icon="📄")
