"""Weighted performance metrics for binary risk predictions.

Every function accepts optional nonnegative weights. With inverse probability
weights restricted to untreated patients, each metric estimates the model's
performance against the outcome patients would have had without the
model-triggered intervention.
"""

from __future__ import annotations

import numpy as np
from scipy.optimize import minimize

_EPS = 1e-12


def _prep(y, p, w=None):
    y = np.asarray(y, dtype=float)
    p = np.asarray(p, dtype=float)
    w = np.ones_like(y) if w is None else np.asarray(w, dtype=float)
    if not (y.shape == p.shape == w.shape):
        raise ValueError("y, p and w must have the same shape")
    if np.any(w < 0):
        raise ValueError("weights must be nonnegative")
    keep = w > 0
    return y[keep], p[keep], w[keep]


def observed_expected(y, p, w=None) -> float:
    """Ratio of the weighted observed event rate to the weighted mean prediction.

    1.0 means calibration in the large. Above 1 the model under-predicts.
    """
    y, p, w = _prep(y, p, w)
    expected = np.sum(w * p)
    if expected <= 0:
        return float("nan")
    return float(np.sum(w * y) / expected)


def event_rate(y, w=None) -> float:
    y = np.asarray(y, dtype=float)
    w = np.ones_like(y) if w is None else np.asarray(w, dtype=float)
    total = np.sum(w)
    return float(np.sum(w * y) / total) if total > 0 else float("nan")


def brier(y, p, w=None) -> float:
    y, p, w = _prep(y, p, w)
    if np.sum(w) <= 0:
        return float("nan")
    return float(np.sum(w * (y - p) ** 2) / np.sum(w))


def auc(y, p, w=None) -> float:
    """Weighted area under the ROC curve (weighted Mann-Whitney statistic).

    Ties in the score count one half. Returns NaN when either class has zero
    total weight.
    """
    y, p, w = _prep(y, p, w)
    pos = y == 1
    wp, wn = np.sum(w[pos]), np.sum(w[~pos])
    if wp <= 0 or wn <= 0:
        return float("nan")
    _, group = np.unique(p, return_inverse=True)  # groups sorted by score, ties share a group
    pos_w = np.bincount(group, weights=w * pos)
    neg_w = np.bincount(group, weights=w * ~pos)
    neg_below = np.cumsum(neg_w) - neg_w
    total = np.sum(pos_w * (neg_below + 0.5 * neg_w))
    return float(total / (wp * wn))


def _logit(p):
    p = np.clip(p, 1e-6, 1 - 1e-6)
    return np.log(p / (1 - p))


def calibration_intercept_slope(y, p, w=None) -> tuple[float, float]:
    """Weighted logistic recalibration: logit P(y=1) = a + b * logit(p).

    Returns (a, b). A calibrated model has a = 0 and b = 1. Fitted by
    unpenalised weighted maximum likelihood.
    """
    y, p, w = _prep(y, p, w)
    if len(y) < 3 or np.sum(w[y == 1]) <= 0 or np.sum(w[y == 0]) <= 0:
        return float("nan"), float("nan")
    x = _logit(p)

    def nll(theta):
        z = theta[0] + theta[1] * x
        return np.sum(w * (np.logaddexp(0, z) - y * z))

    def grad(theta):
        z = theta[0] + theta[1] * x
        r = w * (1 / (1 + np.exp(-z)) - y)
        return np.array([np.sum(r), np.sum(r * x)])

    res = minimize(nll, x0=np.array([0.0, 1.0]), jac=grad, method="BFGS")
    return float(res.x[0]), float(res.x[1])


def calibration_in_the_large(y, p, w=None) -> float:
    """Weighted intercept with the slope fixed at 1 (log odds offset)."""
    y, p, w = _prep(y, p, w)
    if np.sum(w[y == 1]) <= 0 or np.sum(w[y == 0]) <= 0:
        return float("nan")
    x = _logit(p)

    def nll(a):
        z = a[0] + x
        return np.sum(w * (np.logaddexp(0, z) - y * z))

    res = minimize(nll, x0=np.array([0.0]), method="BFGS")
    return float(res.x[0])
