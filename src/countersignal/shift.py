"""Label-free early signals: is the incoming case mix different from reference?

These run as soon as patients are scored, before outcomes are recorded. They
say the population changed, not that performance changed.
"""

from __future__ import annotations

import numpy as np


def psi(reference, current, bins: int = 10, eps: float = 1e-4) -> float:
    """Population stability index using reference quantile bins.

    Common rule of thumb: below 0.1 little change, 0.1 to 0.25 moderate,
    above 0.25 large. These cut points are a convention, not a test.
    """
    reference = np.asarray(reference, dtype=float)
    current = np.asarray(current, dtype=float)
    reference = reference[~np.isnan(reference)]
    current = current[~np.isnan(current)]
    if len(reference) == 0 or len(current) == 0:
        return float("nan")
    edges = np.unique(np.quantile(reference, np.linspace(0, 1, bins + 1)))
    if len(edges) < 2:
        return 0.0 if np.all(current == reference[0]) else float("inf")
    edges[0], edges[-1] = -np.inf, np.inf
    r = np.histogram(reference, edges)[0] / len(reference)
    c = np.histogram(current, edges)[0] / len(current)
    r, c = np.clip(r, eps, None), np.clip(c, eps, None)
    return float(np.sum((c - r) * np.log(c / r)))


def psi_label(value: float) -> str:
    if np.isnan(value):
        return "n/a"
    if value < 0.1:
        return "little change"
    if value < 0.25:
        return "moderate change"
    return "large change"
