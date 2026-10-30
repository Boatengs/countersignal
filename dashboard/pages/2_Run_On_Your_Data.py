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


def _numeric(c):
    return pd.api.types.is_numeric_dtype(log[c]) and not pd.api.types.is_bool_dtype(log[c])


def _binary(c, blanks_ok):
    v = log[c]
    if not blanks_ok and v.isna().any():
        return False
    v = v.dropna()
    return len(v) > 0 and _numeric(c) and v.isin([0, 1]).all()


def _probability(c):
    v = log[c].dropna()
    return _numeric(c) and len(v) > 0 and v.between(0, 1).all() and v.nunique() > 2


# only columns that can serve each role are offered, so an invalid choice cannot be made
in_reference = set(reference.columns)
valid = {"Score": [c for c in cols if _probability(c) and c in in_reference],
         "Outcome": [c for c in cols if _binary(c, blanks_ok=True) and c in in_reference],
         "Treated": [c for c in cols if _binary(c, blanks_ok=False)],
         "Month": [c for c in cols if 0 < log[c].nunique() <= 60 and not log[c].isna().any()]}
rules = {"Score": "numeric predicted risks between 0 and 1, present in both files",
         "Outcome": "0/1 values, blank where not yet recorded, present in both files",
         "Treated": "0/1 values with no blanks",
         "Month": "one value per reporting period (at most 60), no blanks"}
empty = [f"{role} ({rules[role]})" for role, options in valid.items() if not options]
if empty:
    st.error("The deployment log has no column suitable for: " + "; ".join(empty) + ".")
    st.stop()


def pick(role, label, default, taken):
    options = [c for c in valid[role] if c not in taken] or valid[role]
    return st.selectbox(label, options, index=options.index(default) if default in options else 0,
                        help=f"Columns with {rules[role]}.")


c1, c2, c3, c4 = st.columns(4)
with c1:
    score = pick("Score", "Score column", "score", [])
with c2:
    outcome = pick("Outcome", "Outcome column", "outcome", [score])
with c3:
    treated = pick("Treated", "Treated column", "treated", [score, outcome])
with c4:
    window = pick("Month", "Month column", "window", [score, outcome, treated])
roles = {"Score": score, "Outcome": outcome, "Treated": treated, "Month": window}
if len(set(roles.values())) < len(roles):
    st.error("Each of the score, outcome, treated and month columns must be a different column.")
    st.stop()
n_windows = log[window].nunique()
if n_windows > 60:
    st.error(f"The month column '{window}' has {n_windows:,} distinct values. Choose the column "
             "that labels the reporting period (for example 2025-01), with one value per month.")
    st.stop()
covariate_options = [c for c in cols if c not in roles.values() and _numeric(c)
                     and not log[c].isna().any()]
covariates = st.multiselect("Covariates that influenced treatment", covariate_options,
                            default=[c for c in ("x1", "x2", "alerted") if c in covariate_options])
lo, hi = st.slider("Acceptable observed/expected range", 0.5, 2.0, (0.8, 1.25), 0.05)
missing = [c for c in [score, outcome, *covariates] if c not in reference.columns]
if missing:
    st.error(f"Reference file is missing: {', '.join(missing)}")
    st.stop()

cfg = MonitorConfig(score=score, outcome=outcome, treated=treated, window=window,
                    covariates=covariates, oe_band=(lo, hi),
                    horizon=max(log[window].nunique(), 1), n_boot=150)
try:
    with st.spinner("Monitoring..."):
        result = Monitor(reference, cfg).run(log)
except (ValueError, KeyError, TypeError) as err:
    st.error(f"CounterSignal could not run with these columns: {err}")
    st.stop()

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
