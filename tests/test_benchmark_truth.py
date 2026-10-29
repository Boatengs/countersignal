"""The benchmark's population truth must match what its sampler actually draws."""

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "benchmarks"))
import run_benchmark as B  # noqa: E402


def _prep():
    rng = np.random.default_rng(0)
    n = 6000
    pool = pd.DataFrame({"number_inpatient": rng.poisson(0.7, n).astype(float)})
    pool["score"] = 1 / (1 + np.exp(-(-2.4 + 0.4 * pool.number_inpatient)))
    pool["outcome"] = (rng.random(n) < pool.score).astype(int)
    pool["alerted"] = (pool.score >= pool.score.quantile(0.9)).astype(int)
    prep = {"dataset": "diabetes", "pool": pool}
    prep["drift_k"] = B.drift_weight(prep, 1.35)
    return prep


def test_sampled_windows_match_population_truth():
    prep = _prep()
    rng = np.random.default_rng(1)
    for scenario in ("case_mix", "high_risk_drift"):
        truth = B.population_truth(prep, scenario)[True]["oe"]
        y = s = 0.0
        for _ in range(400):
            idx = B.sample_window(prep, scenario, True, 2000, rng)
            y += prep["pool"].outcome.to_numpy()[idx].sum()
            s += prep["pool"].score.to_numpy()[idx].sum()
        assert abs(y / s - truth) < 0.01, scenario


def test_drift_is_sized_to_its_target():
    prep = _prep()
    assert abs(B.population_truth(prep, "high_risk_drift")[True]["oe"] - 1.35) < 1e-6
