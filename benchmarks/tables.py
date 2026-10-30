"""Markdown tables generated from results/benchmark_summary.json.

The README and docs/benchmark.md embed the output of this script between
<!-- tables:NAME --> markers, so every figure in them comes from the committed
results. Run `python benchmarks/tables.py --write` to refresh both files, and
`python benchmarks/tables.py --check` (used in CI) to confirm they match.
"""

from __future__ import annotations

import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
CONFIGS = {"diabetes-logistic": "Diabetes, logistic", "diabetes-gbm": "Diabetes, boosting",
           "sepsis-logistic": "Sepsis, logistic", "sepsis-gbm": "Sepsis, boosting"}
METHODS = {"naive": "Standard", "untreated_only": "Untreated only", "feng": "Feng et al.",
           "countersignal": "CounterSignal"}


def pct(x) -> str:
    return "n/a" if x is None else f"{x * 100:.1f}%"


def month(e) -> str:
    return "n/a" if e.get("median_alarm_month") is None else f"{e['median_alarm_month']:.0f}"


def _table(header: list[str], rows: list[list[str]]) -> str:
    lines = ["| " + " | ".join(header) + " |", "|" + "---|" * len(header)]
    lines += ["| " + " | ".join(r) + " |" for r in rows]
    return "\n".join(lines)


def scenario(s, cfg, name):
    c = s["configs"][cfg]
    return c["scenarios"][f"{name}@{c['simulation']['effects'][0]}"]


def alarm_table(s, name) -> str:
    rows = [[label, *(pct(scenario(s, cfg, name)[m]["any_alarm_rate"]) for m in METHODS)]
            for cfg, label in CONFIGS.items()]
    return _table(["Dataset, model", *METHODS.values()], rows)


def drift_table(s) -> str:
    rows = []
    for cfg, label in CONFIGS.items():
        d = scenario(s, cfg, "high_risk_drift")
        rows.append([label, *(f"{pct(d[m]['detected_rate'])} (month {month(d[m])})" for m in METHODS)])
    return _table(["Dataset, model", *METHODS.values()], rows)


def drift_before_table(s) -> str:
    rows = [[label, *(pct(scenario(s, cfg, "high_risk_drift")[m]["alarm_before_shift_rate"])
                      for m in METHODS)] for cfg, label in CONFIGS.items()]
    return _table(["Dataset, model", *METHODS.values()], rows)


def estimate_table(s) -> str:
    views = ("naive", "untreated_only", "countersignal")
    rows = []
    for cfg, label in CONFIGS.items():
        for name, state in (("stable", "unchanged"), ("high_risk_drift", "drift")):
            e = scenario(s, cfg, name)
            rows.append([label, state, *(f"{e[m]['oe_abs_error']:.3f}" for m in views),
                         *(pct(e[m]["oe_ci_coverage"]) for m in views)])
    return _table(["Dataset, model", "Scenario", "Error: standard", "Error: untreated only",
                   "Error: CounterSignal", "Coverage: standard", "Coverage: untreated only",
                   "Coverage: CounterSignal"], rows)


def cost_table(s) -> str:
    rows = []
    for cfg, label in CONFIGS.items():
        c = s["configs"][cfg]
        st = scenario(s, cfg, "stable")
        sm = scenario(s, cfg, "small_hospital")
        rows.append([label, f"{c['simulation']['n_per_window']:,}", f"{st['mean_ess']:,.0f}",
                     f"{st['countersignal']['insufficient_windows']:.1f}",
                     f"{st['untreated_only']['insufficient_windows']:.1f}",
                     f"{c['simulation']['n_small_hospital']:,}", f"{sm['mean_ess']:,.0f}",
                     f"{sm['countersignal']['insufficient_windows']:.1f}"])
    return _table(["Dataset, model", "Patients a month", "Effective untreated",
                   "Months undecided: CounterSignal", "Months undecided: untreated only",
                   "Small hospital patients a month", "Small hospital effective",
                   "Small hospital months undecided: CounterSignal"], rows)


def feng_ref_table(s) -> str:
    rows = []
    for cfg, label in CONFIGS.items():
        st = scenario(s, cfg, "stable")
        rows.append([label, pct(st["feng"]["any_alarm_rate"]), pct(st["feng_ref"]["any_alarm_rate"])])
    return _table(["Dataset, model", "Calibrated on a run-in under the deployment policy",
                   "Calibrated on the pre-deployment reference split"], rows)


def effect_table(s) -> str:
    c = s["configs"]["diabetes-logistic"]
    rows = []
    for eff in sorted(c["simulation"]["effects"]):
        e = c["scenarios"][f"stable@{eff}"]
        rows.append([f"{eff:.0%}", *(pct(e[m]["any_alarm_rate"]) for m in METHODS)])
    return _table(["Share of events prevented", *METHODS.values()], rows)


def data_table(s) -> str:
    rows = []
    for cfg, label in CONFIGS.items():
        c = s["configs"][cfg]
        t = c["population_truth"]
        rows.append([label, c["model"], f"{c['reference_auc']:.3f}", f"{c['pool_oe']:.3f}",
                     f"{t['case_mix']['after']['oe']:.3f}", f"{t['high_risk_drift']['after']['oe']:.2f}"])
    return _table(["Dataset, model", "Model", "Validation AUC", "True O/E, unchanged",
                   "True O/E, case mix after", "True O/E, drift after"], rows)


TABLES = {"alarms_stable": lambda s: alarm_table(s, "stable"),
          "alarms_case_mix": lambda s: alarm_table(s, "case_mix"),
          "alarms_small": lambda s: alarm_table(s, "small_hospital"),
          "alarms_hidden": lambda s: alarm_table(s, "hidden_confounder"),
          "drift": drift_table, "drift_before": drift_before_table, "estimates": estimate_table,
          "cost": cost_table, "feng_ref": feng_ref_table, "effects": effect_table, "data": data_table}


def render(text: str, s: dict) -> str:
    def sub(m):
        return f"<!-- tables:{m.group(1)} -->\n{TABLES[m.group(1)](s)}\n<!-- /tables -->"
    return re.sub(r"<!-- tables:(\w+) -->.*?<!-- /tables -->", sub, text, flags=re.S)


def main() -> None:
    s = json.loads((ROOT / "results" / "benchmark_summary.json").read_text())
    stale = []
    for path in (ROOT / "README.md", ROOT / "docs" / "benchmark.md"):
        old = path.read_text()
        new = render(old, s)
        if "--write" in sys.argv:
            path.write_text(new)
        elif new != old:
            stale.append(path.name)
    if stale:
        sys.exit(f"tables out of date with results: {stale}; run python benchmarks/tables.py --write")


if __name__ == "__main__":
    main()
