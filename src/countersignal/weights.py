"""Treatment propensity and inverse probability weights.

After deployment, some patients receive an intervention because the model
flagged them. Their observed outcome is no longer the outcome the model was
built to predict. CounterSignal keeps the untreated patients and reweights
them by 1 / P(untreated | covariates) so that, under the assumptions listed in
docs/method.md, they stand in for the whole deployed population.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from .metrics import _logit


def fit_logistic(x: np.ndarray, y: np.ndarray, ridge: float = 1e-6,
                 max_iter: int = 50) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Logistic regression by Newton-Raphson on standardized columns.

    A tiny ridge penalty keeps the fit finite under separation; otherwise it
    is maximum likelihood. Returns (coefficients with intercept first,
    column means, column scales).
    """
    mean = x.mean(axis=0)
    scale = x.std(axis=0)
    scale[scale == 0] = 1.0
    z = np.column_stack([np.ones(len(x)), (x - mean) / scale])
    beta = np.zeros(z.shape[1])
    penalty = np.full(z.shape[1], ridge)
    penalty[0] = 0.0
    for _ in range(max_iter):
        eta = np.clip(z @ beta, -30, 30)
        prob = 1 / (1 + np.exp(-eta))
        grad = z.T @ (y - prob) - penalty * beta
        hess = (z * (prob * (1 - prob))[:, None]).T @ z + np.diag(penalty)
        step = np.linalg.solve(hess, grad)
        beta = beta + step
        if np.max(np.abs(step)) < 1e-8:
            break
    return beta, mean, scale


def predict_logistic(beta: np.ndarray, mean: np.ndarray, scale: np.ndarray,
                     x: np.ndarray) -> np.ndarray:
    eta = np.clip(beta[0] + ((x - mean) / scale) @ beta[1:], -30, 30)
    return 1 / (1 + np.exp(-eta))


@dataclass
class PropensityFit:
    """Fitted model of P(treated = 1 | logit score, covariates)."""

    params: tuple | None
    columns: list[str]
    constant: float | None = None  # used when every row has the same treatment

    def p_untreated_x(self, x: np.ndarray) -> np.ndarray:
        if self.params is None:
            return np.full(len(x), 1.0 - float(self.constant))
        return 1.0 - predict_logistic(*self.params, x)

    def p_untreated(self, frame: pd.DataFrame, score: str) -> np.ndarray:
        return self.p_untreated_x(design(frame, score, self.columns))


def design(frame: pd.DataFrame, score: str, covariates: list[str]) -> np.ndarray:
    cols = [_logit(frame[score].to_numpy(dtype=float))]
    for c in covariates:
        cols.append(frame[c].to_numpy(dtype=float))
    return np.column_stack(cols)


def fit_propensity_x(x: np.ndarray, a: np.ndarray, columns: list[str]) -> PropensityFit:
    a = np.asarray(a, dtype=float)
    if a.min() == a.max():
        return PropensityFit(params=None, columns=list(columns), constant=float(a[0]))
    return PropensityFit(params=fit_logistic(x, a), columns=list(columns))


def fit_propensity(frame: pd.DataFrame, score: str, treated: str,
                   covariates: list[str]) -> PropensityFit:
    """Logistic regression of the logged treatment on the score and covariates.

    The covariates should include everything that influenced who was treated
    (the no unmeasured confounding assumption).
    """
    return fit_propensity_x(design(frame, score, covariates),
                            frame[treated].to_numpy(dtype=int), covariates)


def ipw_weights(treated: np.ndarray, p_untreated: np.ndarray,
                floor: float = 0.02) -> np.ndarray:
    """Weights (1 - A) / max(P(A = 0 | X), floor).

    Treated rows get weight 0. The floor caps any single weight at 1 / floor and
    is reported, because heavy truncation means positivity is in doubt.
    """
    a = np.asarray(treated, dtype=int)
    p0 = np.clip(np.asarray(p_untreated, dtype=float), floor, 1.0)
    return (1 - a) / p0


def effective_sample_size(w: np.ndarray) -> float:
    w = np.asarray(w, dtype=float)
    s2 = np.sum(w ** 2)
    return float(np.sum(w) ** 2 / s2) if s2 > 0 else 0.0
