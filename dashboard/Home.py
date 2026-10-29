"""CounterSignal dashboard: home."""

import pandas as pd
import streamlit as st
from common import CONFIG_LABELS, METHOD_LABELS, page_setup, pct, pick_config, summary

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

st.subheader("Benchmark on real patient records")
st.markdown(
    "Two public datasets, each with two risk models: 30 day readmission among diabetic inpatients "
    "in 130 U.S. hospitals (UCI Diabetes 130-US Hospitals, 1999 to 2008) and in hospital death "
    "among sepsis admissions in Norway (UCI Sepsis Survival, 2011 to 2012). Deployment is "
    "simulated on top of the real outcomes: alerted patients are treated and treatment prevents a "
    "share of their events. Each month is drawn from a known population, so the model's true "
    "calibration in every month is known and every monitor can be scored against it.")


def first_month(e):
    return "n/a" if e.get("median_alarm_month") is None else f"{e['median_alarm_month']:.0f}"


overview = []
for name, label in CONFIG_LABELS.items():
    c = s["configs"].get(name)
    if not c:
        continue
    main = c["simulation"]["effects"][0]
    st_ = c["scenarios"].get(f"stable@{main}")
    dr = c["scenarios"].get(f"high_risk_drift@{main}")
    for m, mlabel in METHOD_LABELS.items():
        overview.append({"Dataset and model": label, "Monitor": mlabel,
                         "False alarm, model unchanged": pct(st_[m]["any_alarm_rate"]) if st_ else "n/a",
                         "Drift caught": pct(dr[m].get("detected_rate")) if dr else "n/a",
                         "Median month of first alarm": first_month(dr[m]) if dr else "n/a"})
st.markdown("**All four configurations** (planned false alarm rate 5%; drift begins in month 5)")
st.dataframe(pd.DataFrame(overview), hide_index=True, width="stretch", height=36 * (len(overview) + 1) + 4)

st.subheader("One configuration in detail")
name = pick_config(s)
c = s["configs"][name]
sim = c["simulation"]
main = sim["effects"][0]
sc = {k.split("@")[0]: v for k, v in c["scenarios"].items() if float(k.split("@")[1]) == main}
truth = c["population_truth"]
st.markdown(
    f"{c['split']['pool'] + c['split']['reference'] + c['split']['development']:,} records, "
    f"{c['model']} predicting {c['outcome']} (AUC {c['reference_auc']:.3f} at validation). "
    f"{sim['n_per_window']:,} patients a month, {sim['replicates']} simulated deployments per "
    f"scenario, treatment prevents {main:.0%} of events and is more likely with higher "
    f"{sim['treatment_driver']}.")

rows = []
for key in ("stable", "case_mix", "small_hospital"):
    if key in sc:
        t = truth[key]
        note = (f"true O/E {t['before']['oe']:.2f}" if key != "case_mix" else
                f"true O/E {t['before']['oe']:.2f} then {t['after']['oe']:.2f}")
        rows.append({"Scenario": sc[key]["label"], "Truth": note,
                     **{METHOD_LABELS[m]: pct(sc[key][m]["any_alarm_rate"]) for m in METHOD_LABELS}})
st.markdown("**Alarm rate when the model itself has not changed** (share of 12 month deployments "
            "with any alarm)")
st.dataframe(pd.DataFrame(rows), hide_index=True, width="stretch")

if "high_risk_drift" in sc:
    d = sc["high_risk_drift"]
    t = truth["high_risk_drift"]
    st.markdown(f"**Detecting real drift:** {d['label'].lower()} (true O/E {t['before']['oe']:.2f} "
                f"before, {t['after']['oe']:.2f} after)")
    measures = [
        ("Deployments where the change was caught", lambda e: pct(e.get("detected_rate"))),
        ("Alarm before the change began", lambda e: pct(e.get("alarm_before_shift_rate"))),
        ("Median month of first alarm", first_month),
    ]
    st.dataframe(pd.DataFrame([{"Measure": n, **{METHOD_LABELS[m]: f(d[m]) for m in METHOD_LABELS}}
                               for n, f in measures]), hide_index=True, width="stretch")

st.caption(
    "The standard, untreated only and CounterSignal monitors share one cumulative alarm rule and "
    "limit and test whether the model is calibrated. The Feng et al. monitor is an independent "
    "implementation of their score based CUSUM: it tests whether outcomes among untreated patients "
    "change relative to the start, given the score and the treatment confounders. In the case mix "
    "scenario the model's own calibration moves slightly, so not every alarm there is false.")

with st.expander("What the correction costs"):
    e = sc["stable"]
    st.markdown(
        f"Reweighting leaves about {e['mean_ess']:,.0f} effective patients out of "
        f"{sim['n_per_window']:,} a month. With the model unchanged, CounterSignal's monthly "
        f"observed/expected was off from the truth by {e['countersignal']['oe_abs_error']:.3f} on "
        f"average, against {e['untreated_only']['oe_abs_error']:.3f} for the untreated only view, and "
        f"it returned \"not enough evidence yet\" in {e['countersignal']['insufficient_windows']:.1f} "
        f"of 12 months against {e['untreated_only']['insufficient_windows']:.1f}. Its 95% intervals "
        f"contained the true value in {pct(e['countersignal']['oe_ci_coverage'])} of months.")

st.page_link("pages/1_Benchmark_Explorer.py", label="Explore a simulated deployment month by month",
             icon="📈")
st.page_link("pages/2_Run_On_Your_Data.py", label="Run CounterSignal on your own deployment log", icon="🧪")
st.page_link("pages/3_Method.py", label="Method, assumptions and limits", icon="📄")
