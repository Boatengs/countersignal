"""Semi-synthetic benchmark of clinical model monitoring under interventions.

Design (see docs/benchmark.md):
* Real patients and real outcomes from two public datasets. The recorded
  outcome is treated as the untreated outcome, because no model influenced
  their care.
* A risk model (logistic regression or gradient boosting) is trained on 30% of
  patients, its reference performance measured on 20%, and the remaining 50%
  form the pool that monthly deployment windows are drawn from.
* Deployment is simulated: patients above the reference 90th percentile of
  risk are alerted; clinicians treat 70% of alerted and 5% of other patients,
  more often for patients with higher values of one recorded clinical factor;
  treatment prevents a fixed share of events (40% by default). Those numbers
  are assumptions of the benchmark, not estimates from the literature.
* Each window is a draw from a known population (the pool under the
  scenario's sampling weights), so the true calibration and discrimination of
  the model in every window is known exactly and each method can be scored.

Run one configuration with --config, then benchmarks/aggregate.py.
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
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from countersignal import DRIFT, INSUFFICIENT, STABLE, Monitor, MonitorConfig  # noqa: E402
from countersignal import metrics as M  # noqa: E402
from countersignal.baselines.feng_score_cusum import FengScoreCUSUM  # noqa: E402
from countersignal.datasets import uci_diabetes, uci_sepsis  # noqa: E402

WINDOWS = 12
BAND = (0.8, 1.25)

DATASETS = {
    "diabetes": dict(
        module=uci_diabetes, archive="data/raw/diabetes_130_us_hospitals.zip",
        manifest="data/manifest.json", outcome="30 day readmission",
        driver="number_inpatient", driver_label="prior inpatient stays",
        covariates=["alerted", "number_inpatient", "age"], n=2000, n_small=150,
        case_mix=lambda f: np.exp(0.6 * np.minimum(f["number_inpatient"].to_numpy(float), 6)),
        case_mix_label="Sicker case mix (more prior inpatient stays) from month 5, model unchanged"),
    "sepsis": dict(
        module=uci_sepsis, archive="data/raw/sepsis_survival.zip",
        manifest="data/manifest_sepsis.json", outcome="in hospital death",
        driver="age_years", driver_label="age",
        covariates=["alerted", "age_years", "episode_number"], n=3000, n_small=200,
        case_mix=lambda f: np.exp(0.04 * (f["age_years"].to_numpy(float) - 60)),
        case_mix_label="Older case mix from month 5, model unchanged"),
}

CONFIGS = {
    "diabetes-logistic": dict(dataset="diabetes", model="logistic", effects=(0.4, 0.2, 0.6)),
    "diabetes-gbm": dict(dataset="diabetes", model="gbm", effects=(0.4,)),
    "sepsis-logistic": dict(dataset="sepsis", model="logistic", effects=(0.4,)),
    "sepsis-gbm": dict(dataset="sepsis", model="gbm", effects=(0.4,)),
}

SCENARIOS = ["stable", "case_mix", "high_risk_drift", "small_hospital"]
SHIFT_FROM = {"stable": None, "case_mix": 4, "high_risk_drift": 4, "small_hospital": None}

METHODS = {"naive": "_naive", "untreated_only": "_untreated", "countersignal": ""}
OE_KEYS = {"naive": "oe_naive", "untreated_only": "oe_untreated", "countersignal": "oe_ipw"}
VERDICT_KEYS = {"naive": "naive_oe_verdict", "untreated_only": "untreated_oe_verdict",
                "countersignal": "oe_verdict"}
MODEL_LABELS = {"logistic": "logistic regression", "gbm": "gradient boosting (histogram based)"}


def _sigmoid(z):
    return 1 / (1 + np.exp(-z))


def scenario_label(dataset: str, scenario: str) -> str:
    d = DATASETS[dataset]
    n = d["n_small"] if scenario == "small_hospital" else d["n"]
    return {
        "stable": "Model unchanged, interventions on",
        "case_mix": d["case_mix_label"],
        "high_risk_drift": "Outcome risk rises among alerted patients from month 5",
        "small_hospital": f"Small hospital, {n} patients a month, model unchanged",
    }[scenario]


def fit_model(kind: str, x: pd.DataFrame, y: pd.Series):
    if kind == "logistic":
        model = make_pipeline(StandardScaler(), LogisticRegression(max_iter=5000))
    else:
        model = HistGradientBoostingClassifier(learning_rate=0.05, max_iter=400, early_stopping=True,
                                               validation_fraction=0.15, random_state=0)
    return model.fit(x, y)


def prepare(config: str, seed: int = 2026) -> dict:
    spec = CONFIGS[config]
    d = DATASETS[spec["dataset"]]
    cohort, counts = d["module"].load(ROOT / d["archive"], ROOT / d["manifest"])
    rng = np.random.default_rng(seed)
    order = rng.permutation(len(cohort))
    n = len(cohort)
    dev = cohort.iloc[order[: int(0.3 * n)]]
    ref = cohort.iloc[order[int(0.3 * n): int(0.5 * n)]].copy()
    pool = cohort.iloc[order[int(0.5 * n):]].copy()
    features = d["module"].feature_columns(cohort)
    model = fit_model(spec["model"], dev[features], dev["outcome"])
    for frame in (ref, pool):
        frame["score"] = np.clip(model.predict_proba(frame[features])[:, 1], 1e-4, 1 - 1e-4)
    cut = float(np.quantile(ref["score"], 0.9))
    for frame in (ref, pool):
        frame["alerted"] = (frame["score"] >= cut).astype(int)
    driver = d["driver"]
    return {"config": config, "dataset": spec["dataset"], "model": spec["model"], "counts": counts,
            "dev_n": len(dev), "ref": ref.reset_index(drop=True), "pool": pool.reset_index(drop=True),
            "features": features, "alert_cut": cut,
            "driver": (driver, float(ref[driver].mean()), float(ref[driver].std()))}


def row_probabilities(prep: dict, scenario: str, active: bool) -> np.ndarray:
    """Probability that each pool row is drawn into a window: the window's population."""
    pool = prep["pool"]
    n = len(pool)
    if not active or SHIFT_FROM[scenario] is None:
        return np.full(n, 1 / n)
    if scenario == "case_mix":
        w = DATASETS[prep["dataset"]]["case_mix"](pool)
        return w / w.sum()
    alerted = pool["alerted"].to_numpy() == 1
    positive = pool["outcome"].to_numpy() == 1
    share = alerted.mean()
    q = np.zeros(n)
    wa = np.where(positive[alerted], 3.0, 1.0)
    q[alerted] = share * wa / wa.sum()
    q[~alerted] = (1 - share) / (~alerted).sum()
    return q


def population_truth(prep: dict, scenario: str) -> dict:
    out = {}
    pool = prep["pool"]
    for active in (False, True):
        q = row_probabilities(prep, scenario, active)
        y, s = pool["outcome"].to_numpy(float), pool["score"].to_numpy(float)
        out[active] = {"oe": float(np.sum(q * y) / np.sum(q * s)), "auc": M.auc(y, s, q)}
    return out


def sample_window(prep, scenario, active, n, rng):
    pool = prep["pool"]
    if active and scenario == "high_risk_drift":
        idx = np.arange(len(pool))
        alerted = pool["alerted"].to_numpy() == 1
        positive = pool["outcome"].to_numpy() == 1
        n_alert = int(round(alerted.mean() * n))
        w = np.where(positive[alerted], 3.0, 1.0)
        pick_a = rng.choice(idx[alerted], n_alert, replace=False, p=w / w.sum())
        pick_b = rng.choice(idx[~alerted], n - n_alert, replace=False)
        return np.concatenate([pick_a, pick_b])
    q = row_probabilities(prep, scenario, active)
    return rng.choice(len(pool), n, replace=False, p=q)


def simulate(prep, scenario, rng, effect=0.4, p_alert=0.7, p_other=0.05, driver_effect=0.5):
    """One deployment. Each window draws distinct patients; a patient can recur across windows."""
    d = DATASETS[prep["dataset"]]
    n = d["n_small"] if scenario == "small_hospital" else d["n"]
    col, mean, sd = prep["driver"]
    frames = []
    for t in range(WINDOWS):
        active = SHIFT_FROM[scenario] is not None and t >= SHIFT_FROM[scenario]
        w = prep["pool"].loc[sample_window(prep, scenario, active, n, rng)].copy()
        base = np.where(w["alerted"] == 1, p_alert, p_other)
        logit = np.log(base / (1 - base)) + driver_effect * (w[col] - mean) / sd
        w["treated"] = (rng.random(len(w)) < _sigmoid(logit)).astype(int)
        prevented = w["treated"].to_numpy() * (rng.random(len(w)) < effect)
        w["y0"] = w["outcome"]
        w["outcome"] = w["y0"] * (1 - prevented)
        w["window"] = f"M{t + 1:02d}"
        w["active"] = active
        frames.append(w)
    return pd.concat(frames, ignore_index=True)


def feng_alarms(prep, log, covariates, seed) -> list[bool]:
    """Alarm path of the Feng et al. score based CUSUM on untreated patients."""
    def feats(frame):
        s = frame["score"].to_numpy(float)
        return np.column_stack([np.log(s / (1 - s)), *(frame[c].to_numpy(float) for c in covariates)])

    mon = FengScoreCUSUM(horizon=WINDOWS, alpha=0.05, seed=seed)
    mon.start(feats(prep["ref"]), prep["ref"]["outcome"].to_numpy(float))
    alarms = []
    for _, g in log.groupby("window", sort=False):
        u = g[g["treated"] == 0]
        alarms.append(mon.update(feats(u), u["outcome"].to_numpy(float))["alarm"] or mon.alarmed)
    return alarms


def one_run(args):
    prep, scenario, seed, effect, n_boot, truth = args
    d = DATASETS[prep["dataset"]]
    rng = np.random.default_rng(seed)
    log = simulate(prep, scenario, rng, effect=effect)
    cfg = MonitorConfig(covariates=d["covariates"], horizon=WINDOWS, n_boot=n_boot, seed=seed)
    ref = prep["ref"][["score", "outcome", *d["covariates"]]]
    table = Monitor(ref, cfg).run(log).table
    active = log.groupby("window", sort=False)["active"].first().to_numpy()
    table["true_oe"] = [truth[a]["oe"] for a in active]
    table["true_auc"] = [truth[a]["auc"] for a in active]
    table["true_outside_band"] = (table["true_oe"] < BAND[0]) | (table["true_oe"] > BAND[1])
    table["cusum_alarm_feng"] = feng_alarms(prep, log, d["covariates"], seed)
    shift_from = SHIFT_FROM[scenario]
    summary = {"config": prep["config"], "scenario": scenario, "seed": seed, "effect": effect,
               "mean_ess": float(table["ess"].mean()),
               "true_outside_band_windows": int(table["true_outside_band"].sum())}
    for method, suffix in {**METHODS, "feng": "_feng"}.items():
        alarms = table[f"cusum_alarm{suffix}"].to_numpy(bool)
        summary[f"{method}_any_alarm"] = bool(alarms.any())
        if shift_from is not None:
            summary[f"{method}_alarm_before_shift"] = bool(alarms[:shift_from].any())
            post = alarms[shift_from:]
            summary[f"{method}_delay"] = int(np.argmax(post)) if post.any() else None
        if method == "feng":
            continue
        key = OE_KEYS[method]
        summary[f"{method}_oe_abs_error"] = float(np.mean(np.abs(table[key] - table["true_oe"])))
        summary[f"{method}_oe_ci_coverage"] = float(np.mean(
            (table[f"{key}_lo"] <= table["true_oe"]) & (table["true_oe"] <= table[f"{key}_hi"])))
        verdict = table[VERDICT_KEYS[method]]
        inside = ~table["true_outside_band"]
        summary[f"{method}_false_drift_windows"] = int(((verdict == DRIFT) & inside).sum())
        summary[f"{method}_missed_drift_windows"] = int(((verdict == STABLE) & ~inside).sum())
        summary[f"{method}_insufficient_windows"] = int((verdict == INSUFFICIENT).sum())
    if "auc_ipw" in table:
        summary["countersignal_auc_abs_error"] = float(np.mean(np.abs(table["auc_ipw"] - table["true_auc"])))
        summary["naive_auc_abs_error"] = float(np.mean(np.abs(table["auc_naive"] - table["true_auc"])))
    return summary, table


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True, choices=list(CONFIGS))
    parser.add_argument("--out", default=str(ROOT / "results"))
    parser.add_argument("--replicates", type=int, default=50)
    parser.add_argument("--n-boot", type=int, default=100)
    parser.add_argument("--workers", type=int, default=4)
    args = parser.parse_args()

    out = Path(args.out) / args.config
    out.mkdir(parents=True, exist_ok=True)
    prep = prepare(args.config)
    ref, pool = prep["ref"], prep["pool"]
    truths = {s: population_truth(prep, s) for s in SCENARIOS}

    jobs = []
    for e_i, effect in enumerate(CONFIGS[args.config]["effects"]):
        scenarios = SCENARIOS if e_i == 0 else ["stable"]
        for s_i, scenario in enumerate(scenarios):
            for r in range(args.replicates):
                seed = 100_000 * (e_i + 1) + 10_000 * (s_i + 1) + r
                jobs.append((prep, scenario, seed, effect, args.n_boot, truths[scenario]))

    start = time.time()
    summaries, examples = [], {}
    with ProcessPoolExecutor(args.workers) as ex:
        for summary, table in ex.map(one_run, jobs, chunksize=2):
            summaries.append(summary)
            key = summary["scenario"]
            if summary["effect"] == CONFIGS[args.config]["effects"][0] and key not in examples:
                examples[key] = table  # first replicate of each scenario, not selected
    pd.DataFrame(summaries).to_csv(out / "runs.csv", index=False)
    for key, table in examples.items():
        table.to_csv(out / f"example_{key}.csv", index=False)

    d = DATASETS[prep["dataset"]]
    meta = {
        "config": args.config, "dataset": prep["dataset"], "model": MODEL_LABELS[prep["model"]],
        "outcome": d["outcome"], "data": prep["counts"],
        "split": {"development": prep["dev_n"], "reference": len(ref), "pool": len(pool)},
        "features": prep["features"], "alert_threshold": prep["alert_cut"],
        "reference_auc": M.auc(ref["outcome"], ref["score"]),
        "reference_oe": M.observed_expected(ref["outcome"], ref["score"]),
        "pool_auc": M.auc(pool["outcome"], pool["score"]),
        "pool_oe": M.observed_expected(pool["outcome"], pool["score"]),
        "population_truth": {s: {"before": t[False], "after": t[True]} for s, t in truths.items()},
        "scenario_labels": {s: scenario_label(prep["dataset"], s) for s in SCENARIOS},
        "simulation": {"windows": WINDOWS, "n_per_window": d["n"], "n_small_hospital": d["n_small"],
                       "p_treat_alerted": 0.7, "p_treat_other": 0.05,
                       "treatment_driver": d["driver_label"], "driver_log_odds_per_sd": 0.5,
                       "effects": list(CONFIGS[args.config]["effects"]), "oe_band": list(BAND),
                       "alpha": 0.05, "propensity_covariates": d["covariates"],
                       "replicates": args.replicates, "n_boot": args.n_boot},
        "runtime_seconds": round(time.time() - start, 1),
    }
    (out / "meta.json").write_text(json.dumps(meta, indent=2, default=float))
    print(json.dumps({k: meta[k] for k in ("config", "reference_auc", "pool_oe", "runtime_seconds")},
                     default=float))


if __name__ == "__main__":
    main()
