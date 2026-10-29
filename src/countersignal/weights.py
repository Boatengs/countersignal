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
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from .metrics import _logit


@dataclass
class PropensityFit:
    """Fitted model of P(treated = 1 | logit score, covariates)."""

    model: object | None
    columns: list[str]
    constant: float | None = None  # used when every row has the same treatment

    def p_untreated(self, frame: pd.DataFrame, score: str) -> np.ndarray:
        if self.model is None:
            return np.full(len(frame), 1.0 - float(self.constant))
        x = design(frame, score, self.columns)
        return self.model.predict_proba(x)[:, 0]


def design(frame: pd.DataFrame, score: str, covariates: list[str]) -> np.ndarray:
    cols = [_logit(frame[score].to_numpy(dtype=float))]
    for c in covariates:
        cols.append(frame[c].to_numpy(dtype=float))
    return np.column_stack(cols)


def fit_propensity(frame: pd.DataFrame, score: str, treated: str,
                   covariates: list[str]) -> PropensityFit:
    """Logistic regression of the logged treatment on the score and covariates.

    The covariates should include everything that influenced who was treated
    (the no unmeasured confounding assumption). Uses a very weak penalty so the
    fit is effectively maximum likelihood but stays finite under separation.
    """
    a = frame[treated].to_numpy(dtype=int)
    if a.min() == a.max():
        return PropensityFit(model=None, columns=list(covariates), constant=float(a[0]))
    model = make_pipeline(StandardScaler(), LogisticRegression(C=1e4, max_iter=5000))
    model.fit(design(frame, score, covariates), a)
    return PropensityFit(model=model, columns=list(covariates))


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
