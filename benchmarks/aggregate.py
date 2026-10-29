"""Combine per configuration benchmark outputs into results/benchmark_summary.json."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
METHODS = ["naive", "untreated_only", "countersignal", "feng", "feng_ref"]


def _clean(obj):
    if isinstance(obj, dict):
        return {k: _clean(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_clean(v) for v in obj]
    if isinstance(obj, float) and np.isnan(obj):
        return None
    return obj


def summarize(runs: pd.DataFrame, meta: dict) -> dict:
    scenarios = {}
    for (scenario, effect), g in runs.groupby(["scenario", "effect"]):
        entry = {"replicates": int(len(g)), "label": meta["scenario_labels"][scenario],
                 "mean_ess": float(g["mean_ess"].mean()),
                 "true_outside_band_windows_mean": float(g["true_outside_band_windows"].mean())}
        shift = scenario in ("case_mix", "high_risk_drift")
        for method in METHODS:
            m = {"any_alarm_rate": float(g[f"{method}_any_alarm"].mean())}
            if shift and f"{method}_alarm_before_shift" in g:
                d = g[f"{method}_delay"]
                m["alarm_before_shift_rate"] = float(g[f"{method}_alarm_before_shift"].mean())
                m["detected_rate"] = float(d.notna().mean())
                if d.notna().any():
                    months = d.dropna() + 1
                    m["median_alarm_month"] = float(months.median())
                    m["alarm_month_iqr"] = [float(months.quantile(0.25)), float(months.quantile(0.75))]
                else:
                    m["median_alarm_month"] = None
            if not method.startswith("feng"):
                for key in ("oe_abs_error", "oe_ci_coverage", "false_drift_windows",
                            "missed_drift_windows", "insufficient_windows"):
                    m[key] = float(g[f"{method}_{key}"].mean())
            entry[method] = m
        if "countersignal_auc_abs_error" in g:
            entry["countersignal"]["auc_abs_error"] = float(g["countersignal_auc_abs_error"].mean())
            entry["naive"]["auc_abs_error"] = float(g["naive_auc_abs_error"].mean())
        scenarios[f"{scenario}@{effect}"] = entry
    return scenarios


def main(results: Path = ROOT / "results") -> None:
    configs = {}
    for meta_path in sorted(results.glob("*/meta.json")):
        meta = json.loads(meta_path.read_text())
        runs = pd.read_csv(meta_path.parent / "runs.csv")
        meta["scenarios"] = summarize(runs, meta)
        configs[meta["config"]] = meta
    if not configs:
        sys.exit("no benchmark outputs found")
    out = {"configs": configs, "methods": {
        "naive": "Standard monitor: predictions against every patient's recorded outcome",
        "untreated_only": "Untreated patients only, no reweighting",
        "countersignal": "CounterSignal: inverse probability weighted untreated patients",
        "feng": "Score based CUSUM of Feng et al. on untreated patients, calibrated on a run-in "
                "under the deployment policy (independent implementation)",
        "feng_ref": "The same Feng et al. monitor calibrated on the pre-deployment reference data"}}
    (results / "benchmark_summary.json").write_text(json.dumps(_clean(out), indent=2))
    print(f"wrote summary for {len(configs)} configurations")


if __name__ == "__main__":
    main()
