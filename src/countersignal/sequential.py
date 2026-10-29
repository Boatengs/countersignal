"""Sequential calibration monitoring with exact small-sample standardization.

Each window contributes one statistic: the inverse probability weighted sum of
(observed minus predicted) over untreated patients. Under the null hypothesis
that the model is calibrated for the untreated outcome, each untreated outcome
is Bernoulli(score) given its covariates, so the window statistic's null
distribution can be simulated exactly from that window's own patients. The
simulated p-value is mapped to a standard normal score, which keeps the CUSUM
honest for small hospitals where normal approximations fail. A two-sided
CUSUM then accumulates evidence across windows, with a control limit chosen by
simulation so the chance of any false alarm over the monitoring horizon is
alpha.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.stats import norm


def window_statistic(y, p, w) -> float:
    y, p, w = (np.asarray(v, dtype=float) for v in (y, p, w))
    return float(np.sum(w * (y - p)))


def exact_z(y, p, w, rng: np.random.Generator, n_sim: int = 2000) -> float:
    """Normal score of the observed statistic against its simulated null.

    Uses the mid-p rank so discrete statistics are not systematically
    conservative. Returns NaN when the window has no untreated patients.
    """
    y, p, w = (np.asarray(v, dtype=float) for v in (y, p, w))
    keep = w > 0
    y, p, w = y[keep], p[keep], w[keep]
    if len(y) == 0:
        return float("nan")
    observed = np.sum(w * (y - p))
    sims = rng.random((n_sim, len(p))) < p
    null = (sims * w).sum(axis=1) - np.sum(w * p)
    below = np.mean(null < observed)
    equal = np.mean(np.isclose(null, observed))
    u = np.clip(below + 0.5 * equal, 0.5 / n_sim, 1 - 0.5 / n_sim)
    return float(norm.ppf(u))


@dataclass
class CusumState:
    upper: float = 0.0  # evidence the model under-predicts (O/E > 1)
    lower: float = 0.0  # evidence the model over-predicts (O/E < 1)


def cusum_path(z: list[float], k: float = 0.5) -> list[CusumState]:
    state = CusumState()
    path = []
    for value in z:
        if np.isnan(value):
            path.append(CusumState(state.upper, state.lower))
            continue
        state = CusumState(max(0.0, state.upper + value - k),
                           max(0.0, state.lower - value - k))
        path.append(state)
    return path


def control_limit(horizon: int, alpha: float = 0.05, k: float = 0.5,
                  n_sim: int = 20000, seed: int = 0) -> float:
    """Limit h with P(max over horizon of max(upper, lower) > h) = alpha under N(0,1) inputs."""
    rng = np.random.default_rng(seed)
    z = rng.standard_normal((n_sim, horizon))
    up = np.zeros(n_sim)
    lo = np.zeros(n_sim)
    peak = np.zeros(n_sim)
    for t in range(horizon):
        up = np.maximum(0.0, up + z[:, t] - k)
        lo = np.maximum(0.0, lo - z[:, t] - k)
        peak = np.maximum(peak, np.maximum(up, lo))
    return float(np.quantile(peak, 1 - alpha))
