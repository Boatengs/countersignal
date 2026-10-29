"""Semi-synthetic benchmark on the UCI Diabetes 130-US Hospitals records.

Design (see docs/benchmark.md):
* Real patients and real 30 day readmission outcomes. The recorded outcome is
  treated as the untreated outcome, because no model influenced their care.
* A readmission risk model is trained on 30% of patients, its reference
  performance measured on 20%, and the remaining 50% form the pool that
  monthly deployment windows are drawn from.
* Deployment is simulated: patients above the reference 90th percentile of
  risk are alerted; clinicians treat 70% of alerted and 5% of other patients,
  more often when the patient had more prior inpatient stays; treatment
  prevents a fixed share of readmissions (40% by default). Those numbers are
  assumptions of the benchmark, not estimates from the literature.
* Because the untreated outcome is known, the true performance of the model in
  every window is known, and each monitoring method can be scored.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from countersignal import DRIFT, STABLE, Monitor, MonitorConfig  # noqa: E402
from countersignal import metrics as M  # noqa: E402
from countersignal.datasets import uci_diabetes as U  # noqa: E402

PROPENSITY_COVARIATES = ["alerted", "number_inpatient", "age"]
METHODS = {"naive": "_naive", "untreated_only": "_untreated", "countersignal": ""}
OE_KEYS = {"naive": "oe_naive", "untreated_only": "oe_untreated", "countersignal": "oe_ipw"}
VERDICT_KEYS = {"naive": "naive_oe_verdict", "untreated_only": "untreated_oe_verdict",
                "countersignal": "oe_verdict"}

SCENARIOS = {
    "stable": dict(n=2000, shift_from=None, kind=None,
                   label="Model unchanged, interventions on"),
    "case_mix": dict(n=2000, shift_from=4, kind="case_mix",
                     label="Sicker case mix from month 5, model unchanged"),
    "high_risk_drift": dict(n=2000, shift_from=4, kind="high_risk",
                            label="Readmissions rise among alerted patients from month 5"),
    "small_hospital": dict(n=150, shift_from=None, kind=None,
                           label="Small hospital, 150 patients a month, model unchanged"),
}


def _sigmoid(z):
    return 1 / (1 + np.exp(-z))


def prepare(cohort: pd.DataFrame, seed: int = 2026) -> dict:
    rng = np.random.default_rng(seed)
    order = rng.permutation(len(cohort))
    n = len(cohort)
    dev = cohort.iloc[order[: int(0.3 * n)]]
    ref = cohort.iloc[order[int(0.3 * n): int(0.5 * n)]].copy()
    pool = cohort.iloc[order[int(0.5 * n):]].copy()
    features = U.feature_columns(cohort)
    model = make_pipeline(StandardScaler(), LogisticRegression(max_iter=5000))
    model.fit(dev[features], dev["outcome"])
    for frame in (ref, pool):
        frame["score"] = model.predict_proba(frame[features])[:, 1]
    cut = float(np.quantile(ref["score"], 0.9))
    for frame in (ref, pool):
        frame["alerted"] = (frame["score"] >= cut).astype(int)
    inpatient_mean = float(ref["number_inpatient"].mean())
    inpatient_sd = float(ref["number_inpatient"].std())
    return {"dev_n": len(dev), "ref": ref.reset_index(drop=True), "pool": pool.reset_index(drop=True),
            "features": features, "alert_cut": cut, "inp": (inpatient_mean, inpatient_sd),
            "coefficients": dict(zip(features, model[-1].coef_[0].round(4).tolist(), strict=True))}


def sample_window(pool, available, n, kind, active, rng):
    idx = np.flatnonzero(available)
    weights = np.ones(len(idx))
    if active and kind == "case_mix":
        z = pool.loc[idx, "number_inpatient"].to_numpy(float)
        weights = np.exp(0.6 * np.minimum(z, 6))
    if active and kind == "high_risk":
        alerted = pool.loc[idx, "alerted"].to_numpy() == 1
        positive = pool.loc[idx, "outcome"].to_numpy() == 1
        share = alerted.mean()
        n_alert = int(round(share * n))
        pick_a = rng.choice(idx[alerted], n_alert, replace=False,
                            p=_norm(np.where(positive[alerted], 3.0, 1.0)))
        pick_b = rng.choice(idx[~alerted], n - n_alert, replace=False)
        return np.concatenate([pick_a, pick_b])
    return rng.choice(idx, n, replace=False, p=_norm(weights))


def _norm(w):
    w = np.asarray(w, float)
    return w / w.sum()


def simulate(prep, scenario, rng, effect=0.4, windows=12, p_alert=0.7, p_other=0.05,
             inpatient_effect=0.5):
    pool = prep["pool"]
    spec = SCENARIOS[scenario]
    # Each window draws distinct patients from the full pool. A patient can
    # appear in more than one window, which keeps drift scenarios from running
    # out of high risk readmissions late in the stream.
    available = np.ones(len(pool), bool)
    frames = []
    for t in range(windows):
        active = spec["shift_from"] is not None and t >= spec["shift_from"]
        idx = sample_window(pool, available, spec["n"], spec["kind"], active, rng)
        w = pool.loc[idx].copy()
        mean, sd = prep["inp"]
        base = np.where(w["alerted"] == 1, p_alert, p_other)
        logit = np.log(base / (1 - base)) + inpatient_effect * (w["number_inpatient"] - mean) / sd
        w["treated"] = (rng.random(len(w)) < _sigmoid(logit)).astype(int)
        prevented = w["treated"].to_numpy() * (rng.random(len(w)) < effect)
        w["y0"] = w["outcome"]
        w["outcome"] = w["y0"] * (1 - prevented)
        w["window"] = f"M{t + 1:02d}"
        frames.append(w)
    return pd.concat(frames, ignore_index=True)


def truth_table(log, band):
    rows = []
    for win, g in log.groupby("window", sort=False):
        oe = M.observed_expected(g["y0"], g["score"])
        rows.append({"window": win, "true_oe": oe, "true_auc": M.auc(g["y0"], g["score"]),
                     "true_outside_band": bool(oe < band[0] or oe > band[1])})
    return pd.DataFrame(rows)


def one_run(args):
    prep, scenario, seed, effect, n_boot = args
    rng = np.random.default_rng(seed)
    log = simulate(prep, scenario, rng, effect=effect)
    cfg = MonitorConfig(covariates=PROPENSITY_COVARIATES, horizon=12, n_boot=n_boot, seed=seed)
    ref = prep["ref"][["score", "outcome", *PROPENSITY_COVARIATES]]
    result = Monitor(ref, cfg).run(log)
    table = result.table.merge(truth_table(log, cfg.oe_band), on="window")
    shift_from = SCENARIOS[scenario]["shift_from"]
    summary = {"scenario": scenario, "seed": seed, "effect": effect}
    for method, suffix in METHODS.items():
        alarms = table[f"cusum_alarm{suffix}"].to_numpy()
        first = int(np.argmax(alarms)) if alarms.any() else None
        summary[f"{method}_any_alarm"] = bool(alarms.any())
        summary[f"{method}_first_alarm"] = first
        if shift_from is not None:
            summary[f"{method}_alarm_before_shift"] = bool(alarms[:shift_from].any())
            post = alarms[shift_from:]
            summary[f"{method}_delay"] = int(np.argmax(post)) if post.any() else None
        oe = table[OE_KEYS[method]]
        summary[f"{method}_oe_abs_error"] = float(np.mean(np.abs(oe - table["true_oe"])))
        lo, hi = table[f"{OE_KEYS[method]}_lo"], table[f"{OE_KEYS[method]}_hi"]
        summary[f"{method}_oe_ci_coverage"] = float(np.mean((lo <= table["true_oe"]) &
                                                            (table["true_oe"] <= hi)))
        verdict = table[VERDICT_KEYS[method]]
        inside = ~table["true_outside_band"]
        summary[f"{method}_false_drift_windows"] = int(((verdict == DRIFT) & inside).sum())
        summary[f"{method}_missed_drift_windows"] = int(((verdict == STABLE) & ~inside).sum())
        summary[f"{method}_insufficient_windows"] = int((verdict == "not enough evidence yet").sum())
    summary["true_outside_band_windows"] = int(table["true_outside_band"].sum())
    summary["mean_ess"] = float(table["ess"].mean())
    return summary, table


def _clean(obj):
    """Replace NaN with None so the summary is valid JSON."""
    if isinstance(obj, dict):
        return {k: _clean(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_clean(v) for v in obj]
    if isinstance(obj, float) and np.isnan(obj):
        return None
    return obj


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default=str(ROOT / "data/raw/diabetes_130_us_hospitals.zip"))
    parser.add_argument("--out", default=str(ROOT / "results"))
    parser.add_argument("--replicates", type=int, default=50)
    parser.add_argument("--n-boot", type=int, default=100)
    parser.add_argument("--workers", type=int, default=4)
    args = parser.parse_args()

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    cohort, counts = U.load(args.data, ROOT / "data/manifest.json")
    prep = prepare(cohort)
    ref_auc = M.auc(prep["ref"]["outcome"], prep["ref"]["score"])
    ref_oe = M.observed_expected(prep["ref"]["outcome"], prep["ref"]["score"])
    pool_auc = M.auc(prep["pool"]["outcome"], prep["pool"]["score"])
    pool_oe = M.observed_expected(prep["pool"]["outcome"], prep["pool"]["score"])

    jobs = []
    for s_i, scenario in enumerate(SCENARIOS):
        for r in range(args.replicates):
            jobs.append((prep, scenario, 10_000 * (s_i + 1) + r, 0.4, args.n_boot))
    for e_i, effect in enumerate((0.2, 0.6)):
        for r in range(args.replicates):
            jobs.append((prep, "stable", 90_000 + 1000 * e_i + r, effect, args.n_boot))

    start = time.time()
    summaries, examples = [], {}
    with ProcessPoolExecutor(args.workers) as pool:
        for summary, table in pool.map(one_run, jobs, chunksize=2):
            summaries.append(summary)
            key = summary["scenario"]
            if summary["effect"] == 0.4 and key not in examples:
                examples[key] = table
    runs = pd.DataFrame(summaries)
    runs.to_csv(out / "benchmark_runs.csv", index=False)
    for key, table in examples.items():
        table.to_csv(out / f"example_{key}.csv", index=False)

    agg = {}
    for (scenario, effect), g in runs.groupby(["scenario", "effect"]):
        entry = {"replicates": int(len(g)), "label": SCENARIOS[scenario]["label"],
                 "true_outside_band_windows_mean": float(g["true_outside_band_windows"].mean()),
                 "mean_ess": float(g["mean_ess"].mean())}
        for method in METHODS:
            m = {"any_alarm_rate": float(g[f"{method}_any_alarm"].mean()),
                 "oe_abs_error": float(g[f"{method}_oe_abs_error"].mean()),
                 "oe_ci_coverage": float(g[f"{method}_oe_ci_coverage"].mean()),
                 "false_drift_windows_mean": float(g[f"{method}_false_drift_windows"].mean()),
                 "missed_drift_windows_mean": float(g[f"{method}_missed_drift_windows"].mean()),
                 "insufficient_windows_mean": float(g[f"{method}_insufficient_windows"].mean())}
            if SCENARIOS[scenario]["shift_from"] is not None:
                d = g[f"{method}_delay"]
                m["alarm_before_shift_rate"] = float(g[f"{method}_alarm_before_shift"].mean())
                m["detected_rate"] = float(d.notna().mean())
                m["median_delay_windows"] = float(d.dropna().median()) if d.notna().any() else None
            entry[method] = m
        agg[f"{scenario}@{effect}"] = entry

    report = {
        "data": counts,
        "split": {"development": prep["dev_n"], "reference": len(prep["ref"]), "pool": len(prep["pool"])},
        "model": {"type": "logistic regression, standardized features", "features": prep["features"],
                  "coefficients": prep["coefficients"], "alert_threshold": prep["alert_cut"],
                  "reference_auc": ref_auc, "reference_oe": ref_oe,
                  "pool_auc": pool_auc, "pool_oe": pool_oe},
        "simulation": {"windows": 12, "p_treat_alerted": 0.7, "p_treat_other": 0.05,
                       "inpatient_log_odds_per_sd": 0.5, "main_effect": 0.4,
                       "sensitivity_effects": [0.2, 0.6], "oe_band": [0.8, 1.25],
                       "alpha": 0.05, "propensity_covariates": PROPENSITY_COVARIATES,
                       "replicates": args.replicates, "n_boot": args.n_boot},
        "scenarios": agg,
        "runtime_seconds": round(time.time() - start, 1),
    }
    (out / "benchmark_summary.json").write_text(json.dumps(_clean(report), indent=2))
    print(json.dumps(_clean(agg), indent=2))


if __name__ == "__main__":
    main()
