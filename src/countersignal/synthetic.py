"""Synthetic deployments with known ground truth, for tests and examples."""

from __future__ import annotations

import numpy as np
import pandas as pd


def _sigmoid(z):
    return 1 / (1 + np.exp(-z))


def make_deployment(n_per_window: int = 2000, windows: int = 12, *, seed: int = 0,
                    odds_shift_from: int | None = None, odds_ratio: float = 1.0,
                    alert_quantile: float = 0.9, p_treat_alert: float = 0.7,
                    p_treat_other: float = 0.05, effect: float = 0.5,
                    confounding: float = 0.8,
                    shift_high_risk_only: bool = False) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Return (reference, deployment log).

    The model score equals the true untreated risk at reference time. After
    deployment, patients above the alert quantile are treated with probability
    p_treat_alert and others with p_treat_other, shifted on the log odds scale
    by a covariate (x2) that clinicians also look at. Treatment prevents a
    fraction `effect` of events. From window `odds_shift_from` on, the true
    untreated risk has its odds multiplied by `odds_ratio` while the score is
    unchanged, which is real miscalibration the monitor should catch. With
    shift_high_risk_only the shift applies only to alerted patients, the group
    most likely to be treated and so the hardest to monitor.
    The log keeps the untreated potential outcome in column y0 so tests can
    compute the true performance.
    """
    rng = np.random.default_rng(seed)

    def patients(n):
        x1, x2 = rng.standard_normal(n), rng.standard_normal(n)
        score = _sigmoid(-2.2 + 1.0 * x1 + 0.6 * x2)
        return x1, x2, score

    x1, x2, score = patients(4 * n_per_window)
    cut = np.quantile(score, alert_quantile)
    reference = pd.DataFrame({"x1": x1, "x2": x2, "alerted": (score >= cut).astype(int),
                              "score": score,
                              "outcome": (rng.random(len(score)) < score).astype(int)})
    frames = []
    for t in range(windows):
        x1, x2, score = patients(n_per_window)
        logit = np.log(score / (1 - score))
        alerted = (score >= cut).astype(int)
        if odds_shift_from is not None and t >= odds_shift_from:
            hit = alerted if shift_high_risk_only else np.ones(n_per_window)
            logit = logit + hit * np.log(odds_ratio)
        risk0 = _sigmoid(logit)
        y0 = (rng.random(n_per_window) < risk0).astype(int)
        base = np.where(alerted == 1, p_treat_alert, p_treat_other)
        p_treat = _sigmoid(np.log(base / (1 - base)) + confounding * x2)
        treated = (rng.random(n_per_window) < p_treat).astype(int)
        prevented = treated * (rng.random(n_per_window) < effect)
        y = y0 * (1 - prevented)
        frames.append(pd.DataFrame({"window": f"W{t + 1:02d}", "x1": x1, "x2": x2,
                                    "alerted": alerted, "score": score, "treated": treated,
                                    "outcome": y, "y0": y0, "risk0": risk0}))
    return reference, pd.concat(frames, ignore_index=True)
