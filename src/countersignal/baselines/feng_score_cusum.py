"""Score-based CUSUM of Feng et al. for monitoring under confounding interventions.

An independent implementation, written from the description in

    Feng J, Gossmann A, Pennello G, Petrick N, Sahiner B, Pirracchio R.
    Monitoring machine learning (ML) based risk prediction algorithms in the
    presence of confounding medical interventions. arXiv:2211.09781.

and the logic of the authors' companion code
(https://github.com/jjfeng/monitoring_ML_performativity, which carries no
license, so none of it is copied here). It is included only as a comparison
method for the benchmark.

The procedure uses standard of care (untreated) patients only. A logistic
recalibration model of the outcome on the model's logit and the treatment
confounders is fitted on the pre change data and refitted on all data seen so
far. Each batch adds the score of a shift parameter on the same features. The
chart statistic is the largest L2 norm of the cumulative shift score over all
candidate change points. Control limits come from a parametric bootstrap of
particles under the no change hypothesis, including a first order correction
for estimating the recalibration parameters, with the false alarm budget
spent linearly over the monitoring horizon. Particles that would already have
alarmed are dropped, so each limit is conditional on no earlier alarm.

Departure from the companion code: the estimation correction uses
theta_hat - theta = (-H)^{-1} S (H the Hessian of the log likelihood, S the
cumulative score) at every step. The companion code applies the opposite sign
to the calibration data's term only.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from ..weights import fit_logistic


def _aug(x: np.ndarray) -> np.ndarray:
    return np.column_stack([x, np.ones(len(x))])


def _prob(xa: np.ndarray, theta: np.ndarray) -> np.ndarray:
    return 1 / (1 + np.exp(-np.clip(xa @ theta, -30, 30)))


def _theta_from_fit(beta, mean, scale) -> np.ndarray:
    """Convert a standardized fit (intercept first) to raw coefficients with the intercept last."""
    coef = beta[1:] / scale
    intercept = beta[0] - np.sum(coef * mean)
    return np.append(coef, intercept)


@dataclass
class FengScoreCUSUM:
    """Run with ``start(calibration_x, calibration_y)`` then ``update(batch_x, batch_y)`` per batch.

    x holds the monitoring features for untreated patients: the model's logit
    followed by the treatment confounders.
    """

    horizon: int
    alpha: float = 0.05
    particles_per_batch: int = 4
    seed: int = 0
    history: list = field(default_factory=list)

    def start(self, x_cal: np.ndarray, y_cal: np.ndarray) -> None:
        self.rng = np.random.default_rng(self.seed)
        self.n_particles_max = int(self.particles_per_batch / (self.alpha / self.horizon))
        self.x_all = np.asarray(x_cal, float)
        self.y_all = np.asarray(y_cal, float)
        theta = _theta_from_fit(*fit_logistic(self.x_all, self.y_all))
        xa = _aug(self.x_all)
        p = _prob(xa, theta)
        self.neg_hess = (xa * (p * (1 - p))[:, None]).T @ xa  # minus the Hessian
        # simulated calibration outcomes give each particle its own estimation error
        sim = self.rng.random((self.n_particles_max, len(p))) < p
        self.theta_score_cum = (sim - p) @ xa  # particles x k
        self.inf = [self.theta_score_cum @ np.linalg.inv(self.neg_hess).T]
        k = xa.shape[1]
        self.delta_cusum = np.zeros((self.n_particles_max, 0, k))
        self.correction_cusum = np.zeros((self.n_particles_max, 0, k))
        self.batch_scores: list[np.ndarray] = []  # observed shift score per batch
        self.t = 0
        self.alarmed = False

    def update(self, x: np.ndarray, y: np.ndarray) -> dict:
        x = np.asarray(x, float)
        y = np.asarray(y, float)
        theta = _theta_from_fit(*fit_logistic(self.x_all, self.y_all))  # pre batch pooled fit
        xa = _aug(x)
        p = _prob(xa, theta)

        # observed chart statistic over all candidate change points
        self.batch_scores.append(((y - p)[:, None] * xa).sum(axis=0))
        sums = np.cumsum(np.array(self.batch_scores)[::-1], axis=0)[::-1]
        chart = float(np.max(np.linalg.norm(sums, axis=1)))

        # particles: simulate this batch under no change
        n = self.theta_score_cum.shape[0]
        sim = self.rng.random((n, len(p))) < p
        resid = sim - p  # particles x rows
        batch_score = resid @ xa  # particles x k (same features for theta and shift)
        grad_exp = -(xa * (p * (1 - p))[:, None]).T @ xa  # E d(shift score)/d theta
        correction = self.inf[-1] @ grad_exp.T  # uses the estimate before this batch

        self.theta_score_cum = self.theta_score_cum + batch_score
        self.neg_hess = self.neg_hess + (xa * (p * (1 - p))[:, None]).T @ xa
        self.inf.append(self.theta_score_cum @ np.linalg.inv(self.neg_hess).T)
        self.delta_cusum = np.concatenate([self.delta_cusum, np.zeros((n, 1, xa.shape[1]))], axis=1)
        self.correction_cusum = np.concatenate(
            [self.correction_cusum, np.zeros((n, 1, xa.shape[1]))], axis=1)
        self.delta_cusum += batch_score[:, None, :]
        self.correction_cusum += correction[:, None, :]
        part_chart = np.max(np.linalg.norm(self.delta_cusum + self.correction_cusum, axis=2), axis=1)

        self.t += 1
        spend = self.t * self.alpha / self.horizon
        q = min(1.0, self.n_particles_max * (1 - spend) / n)
        limit = float(np.quantile(part_chart, q))
        keep = part_chart <= limit
        self.theta_score_cum = self.theta_score_cum[keep]
        self.inf = [f[keep] for f in self.inf]
        self.delta_cusum = self.delta_cusum[keep]
        self.correction_cusum = self.correction_cusum[keep]

        self.x_all = np.vstack([self.x_all, x])
        self.y_all = np.concatenate([self.y_all, y])
        alarm = chart > limit
        self.alarmed = self.alarmed or alarm
        row = {"chart": chart, "limit": limit, "alarm": bool(alarm), "particles": int(keep.sum())}
        self.history.append(row)
        return row
