import numpy as np
import pytest
from sklearn.metrics import roc_auc_score

from countersignal import metrics as M


def test_weighted_auc_matches_sklearn():
    rng = np.random.default_rng(1)
    y = rng.integers(0, 2, 500)
    p = np.round(rng.random(500), 2)  # include ties
    w = rng.random(500) * 3
    assert M.auc(y, p, w) == pytest.approx(roc_auc_score(y, p, sample_weight=w), abs=1e-12)
    assert M.auc(y, p) == pytest.approx(roc_auc_score(y, p), abs=1e-12)


def test_oe_and_calibration_of_perfect_model():
    rng = np.random.default_rng(2)
    p = rng.uniform(0.05, 0.6, 200_000)
    y = (rng.random(len(p)) < p).astype(int)
    assert M.observed_expected(y, p) == pytest.approx(1.0, abs=0.01)
    a, b = M.calibration_intercept_slope(y, p)
    assert a == pytest.approx(0.0, abs=0.03)
    assert b == pytest.approx(1.0, abs=0.03)


def test_zero_weight_rows_are_ignored():
    y = np.array([1, 0, 1, 0])
    p = np.array([0.9, 0.1, 0.2, 0.8])
    w = np.array([1, 1, 0, 0])
    assert M.auc(y, p, w) == 1.0
    assert np.isnan(M.auc([1, 1], [0.2, 0.3]))


def test_newton_logistic_matches_sklearn():
    from sklearn.linear_model import LogisticRegression

    from countersignal.weights import fit_logistic, predict_logistic

    rng = np.random.default_rng(3)
    x = rng.normal(size=(3000, 3)) * [1, 5, 0.2] + [0, 10, 1]
    y = (rng.random(3000) < 1 / (1 + np.exp(-(0.4 + x[:, 0] - 0.1 * x[:, 1])))).astype(int)
    ours = predict_logistic(*fit_logistic(x, y), x)
    ref = LogisticRegression(C=np.inf, max_iter=10000).fit(x, y).predict_proba(x)[:, 1]
    def loglik(q):
        return np.sum(y * np.log(q) + (1 - y) * np.log(1 - q))

    assert loglik(ours) >= loglik(ref) - 1e-6  # at least as good a maximum
    assert np.max(np.abs(ours - ref)) < 1e-3


def test_psi_sees_shifts_in_binary_and_discrete_variables():
    from countersignal.shift import psi
    ref = np.r_[np.ones(100), np.zeros(900)]
    cur = np.r_[np.ones(600), np.zeros(400)]
    assert psi(ref, ref) == pytest.approx(0.0, abs=1e-9)
    assert psi(ref, cur) > 0.5
    assert psi(np.arange(1000) % 3, np.zeros(500)) > 0.5
