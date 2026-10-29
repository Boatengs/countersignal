"""Run the monitor on an uploaded deployment log."""

import pandas as pd
import streamlit as st
from common import cusum_chart, oe_chart, page_setup

from countersignal import Monitor, MonitorConfig
from countersignal.report import report_card, to_markdown
from countersignal.synthetic import make_deployment

page_setup("Run On Your Data")
st.title("Run CounterSignal on your data")
st.markdown(
    "Upload two CSV files. **Reference**: the model's validation data from before deployment, "
    "with the score and the outcome. **Deployment log**: one row per scored patient after "
    "deployment, with the month, score, whether they were treated, the outcome (blank if not "
    "recorded yet) and the covariates that influenced who was treated. Files stay in this "
    "session and are not stored.")

use_demo = st.toggle("Use a synthetic demo instead", value=True)
if use_demo:
    reference, log = make_deployment(n_per_window=1500, windows=12, seed=1,
                                     odds_shift_from=6, odds_ratio=3.0,
                                     shift_high_risk_only=True)
    log = log.drop(columns=["y0", "risk0"])
    st.caption("Synthetic demo: risk rises among alerted patients from month 7.")
else:
    a, b = st.columns(2)
    ref_file = a.file_uploader("Reference CSV", type="csv")
    log_file = b.file_uploader("Deployment log CSV", type="csv")
    if not (ref_file and log_file):
        st.stop()
    reference, log = pd.read_csv(ref_file), pd.read_csv(log_file)

cols = list(log.columns)


def pick(label, default, options):
    return st.selectbox(label, options, index=options.index(default) if default in options else 0)


c1, c2, c3, c4 = st.columns(4)
with c1:
    score = pick("Score column", "score", cols)
with c2:
    outcome = pick("Outcome column", "outcome", cols)
with c3:
    treated = pick("Treated column", "treated", cols)
with c4:
    window = pick("Month column", "window", cols)
covariates = st.multiselect("Covariates that influenced treatment",
                            [c for c in cols if c not in (score, outcome, treated, window)],
                            default=[c for c in ("x1", "x2", "alerted") if c in cols])
lo, hi = st.slider("Acceptable observed/expected range", 0.5, 2.0, (0.8, 1.25), 0.05)
missing = [c for c in [score, outcome, *covariates] if c not in reference.columns]
if missing:
    st.error(f"Reference file is missing: {', '.join(missing)}")
    st.stop()

cfg = MonitorConfig(score=score, outcome=outcome, treated=treated, window=window,
                    covariates=covariates, oe_band=(lo, hi),
                    horizon=max(log[window].nunique(), 1), n_boot=150)
with st.spinner("Monitoring..."):
    result = Monitor(reference, cfg).run(log)

table = result.table
st.plotly_chart(oe_chart(table, (lo, hi), truth=False), width="stretch")
st.plotly_chart(cusum_chart(table, result.control_limit), width="stretch")
st.subheader("Report card")
for card in report_card(result):
    with st.expander(f"{card['window']}: {card['headline']}"):
        for line in card["lines"]:
            st.markdown(f"- {line}")
st.download_button("Download report card (Markdown)", to_markdown(result), "countersignal_report.md")
st.download_button("Download monthly table (CSV)", table.to_csv(index=False), "countersignal_table.csv")
