"""The CounterSignal monitor: one row of evidence per monitoring window."""

from __future__ import annotations

import json
import warnings
from dataclasses import asdict, dataclass, field

import numpy as np
import pandas as pd

from . import metrics as M
from .sequential import control_limit, cusum_path, exact_z
from .shift import psi
from .weights import (
    design,
    effective_sample_size,
    fit_propensity,
    fit_propensity_x,
    ipw_weights,
    predict_logistic,
)
from .weights import fit_logistic as _fit_logistic

STABLE = "stable"
DRIFT = "drift"
INSUFFICIENT = "not enough evidence yet"
AWAITING = "awaiting outcomes"
NO_UNTREATED = "no untreated patients"


@dataclass
class MonitorConfig:
    score: str = "score"
    outcome: str = "outcome"
    treated: str = "treated"
    window: str = "window"  # labels must sort in time order (e.g. 2026-01, M01)
    covariates: list[str] = field(default_factory=list)
    oe_band: tuple[float, float] = (0.8, 1.25)
    auc_tolerance: float = 0.05
    alpha: float = 0.05
    horizon: int = 12  # months the alarm limit covers
    cusum_k: float = 0.5
    n_boot: int = 200
    weight_floor: float = 0.02
    bootstrap_refit: bool = True  # refit the treatment model in every bootstrap resample
    min_labelled_share: float = 1.0  # evaluate a window only once this share of outcomes is recorded
    seed: int = 0


def _band_verdict(lo: float, hi: float, band: tuple[float, float]) -> str:
    if np.isnan(lo) or np.isnan(hi):
        return INSUFFICIENT
    if hi < band[0] or lo > band[1]:
        return DRIFT
    if lo >= band[0] and hi <= band[1]:
        return STABLE
    return INSUFFICIENT


def _oe_window_needed(oe: float, lo: float, hi: float, n_eff: float,
                      band: tuple[float, float]) -> float:
    """Effective sample size of a single window whose interval would just clear the nearer band edge.

    Assumes the interval width shrinks with the square root of the effective
    sample size and the point estimate stays where it is. At that size the
    interval only reaches the edge, so a window of this size settles the
    question about half the time. Returns NaN when the estimate sits on an edge.
    """
    if any(np.isnan(v) for v in (oe, lo, hi)) or oe <= 0 or lo <= 0 or n_eff <= 0:
        return float("nan")
    half = (np.log(hi) - np.log(lo)) / 2
    distance = min(abs(np.log(oe) - np.log(band[0])), abs(np.log(oe) - np.log(band[1])))
    if distance < 1e-3:
        return float("nan")
    return float(np.ceil(n_eff * (half / distance) ** 2))


def _validate(frame: pd.DataFrame, cols: list[str], name: str, binary: list[str],
              may_be_blank: str | None = None, score: str | None = None,
              numeric: tuple[str, ...] = ()) -> None:
    missing = [c for c in cols if c not in frame.columns]
    if missing:
        raise ValueError(f"{name} is missing columns: {missing}")
    bad = [c for c in cols if c != may_be_blank and frame[c].isna().any()]
    if bad:
        raise ValueError(f"{name} has missing values in {bad}")
    for c in (*binary, *numeric, *([score] if score else [])):
        if not pd.api.types.is_numeric_dtype(frame[c]):
            raise ValueError(f"{name}.{c} must be numeric")
    for c in binary:
        vals = frame[c].dropna().unique()
        if not np.isin(vals, [0, 1]).all():
            raise ValueError(f"{name}.{c} must be 0/1 (blank allowed only for outcomes)")
    if score is not None:
        s = frame[score].dropna()
        if not (np.isfinite(s).all() and s.between(0, 1).all()):
            raise ValueError(f"{name}.{score} must be a predicted risk between 0 and 1")


class Monitor:
    """Monitor a deployed clinical risk model for drift, correcting for the
    interventions its own alerts trigger.

    reference: pre-deployment validation data with the model's score and the
    outcome, collected while the model did not influence care.
    """

    def __init__(self, reference: pd.DataFrame, config: MonitorConfig | None = None):
        self.cfg = config or MonitorConfig()
        c = self.cfg
        if c.n_boot < 20:
            raise ValueError("n_boot must be at least 20 for percentile intervals")
        _validate(reference, [c.score, c.outcome, *c.covariates], "reference", [c.outcome],
                  may_be_blank=c.outcome, score=c.score, numeric=tuple(c.covariates))
        ref = reference.dropna(subset=[c.score, c.outcome])
        self.reference = ref
        y, p = ref[c.outcome].to_numpy(float), ref[c.score].to_numpy(float)
        self.ref_auc = M.auc(y, p)
        self.ref_oe = M.observed_expected(y, p)
        self.limit = control_limit(c.horizon, c.alpha, c.cusum_k)  # fixed seed: same limit every run
        self.outcome_params = None
        if c.covariates and len(np.unique(y)) == 2:
            self.outcome_params = _fit_logistic(design(ref, c.score, c.covariates), y)

    def _m0(self, x: np.ndarray, p: np.ndarray) -> np.ndarray:
        """Outcome model for the doubly robust estimate, fitted on the reference period."""
        if self.outcome_params is None:
            return p
        return predict_logistic(*self.outcome_params, x)

    def run(self, log: pd.DataFrame) -> MonitorResult:
        c = self.cfg
        _validate(log, [c.score, c.treated, c.window, c.outcome, *c.covariates], "log",
                  [c.treated, c.outcome], may_be_blank=c.outcome, score=c.score,
                  numeric=tuple(c.covariates))
        roles = [c.score, c.outcome, c.treated, c.window]
        if len(set(roles)) < len(roles) or set(roles) & set(c.covariates):
            raise ValueError("score, outcome, treated, window and covariates must be distinct columns")
        log = log.reset_index(drop=True)
        rngs = [np.random.default_rng(s) for s in np.random.SeedSequence(c.seed).spawn(2)]
        try:
            windows = sorted(pd.unique(log[c.window]))
        except TypeError:
            raise ValueError(f"log.{c.window} mixes value types (for example numbers and text), "
                             "so the months cannot be put in order") from None
        if len(windows) > c.horizon:
            warnings.warn(f"{len(windows)} windows exceed the alarm horizon of {c.horizon}; the "
                          "false alarm rate is only controlled over the first horizon windows",
                          stacklevel=2)
        rows = []
        for i, win in enumerate(windows):
            seen = log[log[c.window].isin(windows[: i + 1])]
            prop = fit_propensity(seen, c.score, c.treated, c.covariates)
            cur = log[log[c.window] == win]
            prev = log[log[c.window].isin(windows[:i])]
            rows.append(self._evaluate(win, cur, prop, rngs, prev))
        table = pd.DataFrame(rows)
        # CounterSignal's CUSUM, plus the same rule applied to the two views a
        # standard monitor would use, for comparison.
        for suffix in ("", "_naive", "_untreated"):
            path = cusum_path(table[f"z{suffix}"].tolist(), c.cusum_k)
            table[f"cusum_upper{suffix}"] = [s.upper for s in path]
            table[f"cusum_lower{suffix}"] = [s.lower for s in path]
            table[f"cusum_alarm{suffix}"] = (
                table[[f"cusum_upper{suffix}", f"cusum_lower{suffix}"]].max(axis=1) > self.limit)
        return MonitorResult(table=table, config=c, ref_auc=self.ref_auc,
                             ref_oe=self.ref_oe, control_limit=self.limit)

    def _evaluate(self, win, cur: pd.DataFrame, prop, rngs, prev: pd.DataFrame) -> dict:
        c = self.cfg
        boot_rng, null_rng = rngs
        out: dict = {"window": win, "n": len(cur), "n_treated": int(cur[c.treated].sum())}
        out["mean_score"] = float(cur[c.score].mean())
        out["score_psi"] = psi(self.reference[c.score], cur[c.score])
        cov_psi = {col: psi(self.reference[col], cur[col]) for col in c.covariates}
        if cov_psi:
            top = max(cov_psi, key=lambda k: cov_psi[k])
            out["top_shift_feature"], out["top_shift_psi"] = top, cov_psi[top]
        labelled = cur[c.outcome].notna().to_numpy()
        out["n_labelled"] = int(labelled.sum())
        empty = {"z": np.nan, "z_naive": np.nan, "z_untreated": np.nan}
        if len(cur) == 0 or labelled.mean() < c.min_labelled_share or not labelled.any():
            out.update({"oe_verdict": AWAITING, "auc_verdict": AWAITING,
                        "naive_oe_verdict": AWAITING, "untreated_oe_verdict": AWAITING, **empty})
            return out
        lab = cur[labelled]
        y = lab[c.outcome].to_numpy(float)
        p = lab[c.score].to_numpy(float)
        a = lab[c.treated].to_numpy(int)
        if (a == 0).sum() == 0:
            out.update({"oe_verdict": NO_UNTREATED, "auc_verdict": NO_UNTREATED,
                        "naive_oe_verdict": NO_UNTREATED, "untreated_oe_verdict": NO_UNTREATED,
                        "ess": 0.0, **empty})
            return out
        x_cur = design(lab, c.score, c.covariates)
        p0 = prop.p_untreated_x(x_cur)
        w = ipw_weights(a, p0, c.weight_floor)
        m0 = self._m0(x_cur, p)
        untreated = (1 - a).astype(float)
        out["ess"] = effective_sample_size(w)
        out["weights_truncated"] = int(np.sum((a == 0) & (p0 < c.weight_floor)))

        if len(prev):
            x_prev = design(prev, c.score, c.covariates)
            a_prev = prev[c.treated].to_numpy(int)
        else:
            x_prev, a_prev = np.empty((0, x_cur.shape[1])), np.empty(0, int)
        unlab = cur[~labelled]
        x_unlab = design(unlab, c.score, c.covariates)
        a_unlab = unlab[c.treated].to_numpy(int)

        def refit_weights(idx):
            """Weights from a treatment model refitted on a resample of every row it was fitted on."""
            j = boot_rng.integers(0, len(x_prev), len(x_prev)) if len(x_prev) else np.empty(0, int)
            k = boot_rng.integers(0, len(x_unlab), len(x_unlab)) if len(x_unlab) else np.empty(0, int)
            fit = fit_propensity_x(np.vstack([x_prev[j], x_unlab[k], x_cur[idx]]),
                                   np.concatenate([a_prev[j], a_unlab[k], a[idx]]), c.covariates)
            return ipw_weights(a[idx], fit.p_untreated_x(x_cur[idx]), c.weight_floor)

        def stats(idx, ww=None):
            ww = w[idx] if ww is None else ww
            yy, pp, mm, uu = y[idx], p[idx], m0[idx], untreated[idx]
            dr_rate = np.mean(mm + ww * (yy - mm))
            return {
                "oe_naive": M.observed_expected(yy, pp),
                "auc_naive": M.auc(yy, pp),
                "oe_untreated": M.observed_expected(yy, pp, uu),
                "auc_untreated": M.auc(yy, pp, uu),
                "oe_ipw": M.observed_expected(yy, pp, ww),
                "auc_ipw": M.auc(yy, pp, ww),
                "brier_ipw": M.brier(yy, pp, ww),
                "oe_dr": float(dr_rate / np.mean(pp)) if np.mean(pp) > 0 else np.nan,
            }

        out.update(stats(np.arange(len(y))))
        out["cal_intercept_ipw"] = M.calibration_in_the_large(y, p, w)  # slope fixed at 1
        out["cal_slope_ipw"] = M.calibration_intercept_slope(y, p, w)[1]
        boots = []
        for _ in range(c.n_boot):
            idx = boot_rng.integers(0, len(y), len(y))
            boots.append(stats(idx, refit_weights(idx) if c.bootstrap_refit else None))
        frame = pd.DataFrame(boots)
        for key in ("oe_naive", "auc_naive", "oe_untreated", "auc_untreated",
                    "oe_ipw", "auc_ipw", "brier_ipw", "oe_dr"):
            vals = frame[key].dropna()
            lo, hi = (np.quantile(vals, [0.025, 0.975]) if len(vals) >= 20 else (np.nan, np.nan))
            out[f"{key}_lo"], out[f"{key}_hi"] = float(lo), float(hi)
        out["oe_verdict"] = _band_verdict(out["oe_ipw_lo"], out["oe_ipw_hi"], c.oe_band)
        floor = self.ref_auc - c.auc_tolerance
        if np.isnan(out["auc_ipw_lo"]):
            out["auc_verdict"] = INSUFFICIENT
        elif out["auc_ipw_hi"] < floor:
            out["auc_verdict"] = DRIFT
        elif out["auc_ipw_lo"] >= floor:
            out["auc_verdict"] = STABLE
        else:
            out["auc_verdict"] = INSUFFICIENT
        out["naive_oe_verdict"] = _band_verdict(out["oe_naive_lo"], out["oe_naive_hi"], c.oe_band)
        out["untreated_oe_verdict"] = _band_verdict(
            out["oe_untreated_lo"], out["oe_untreated_hi"], c.oe_band)
        needed = (_oe_window_needed(out["oe_ipw"], out["oe_ipw_lo"], out["oe_ipw_hi"], out["ess"],
                                    c.oe_band) if out["oe_verdict"] == INSUFFICIENT else np.nan)
        out["oe_window_needed_ess"] = needed
        out["oe_window_needed_patients"] = (float(np.ceil(needed * len(y) / out["ess"]))
                                            if not np.isnan(needed) else np.nan)
        out["z"] = exact_z(y, p, w, null_rng)
        out["z_naive"] = exact_z(y, p, np.ones_like(y), null_rng)
        out["z_untreated"] = exact_z(y, p, untreated, null_rng)
        return out


@dataclass
class MonitorResult:
    table: pd.DataFrame
    config: MonitorConfig
    ref_auc: float
    ref_oe: float
    control_limit: float

    @property
    def first_alarm(self):
        hits = self.table.index[self.table["cusum_alarm"]]
        return None if len(hits) == 0 else self.table.loc[hits[0], "window"]

    def to_json(self) -> str:
        cfg = asdict(self.config)
        cfg["oe_band"] = list(cfg["oe_band"])
        return json.dumps({
            "config": cfg,
            "reference": {"auc": self.ref_auc, "oe": self.ref_oe},
            "control_limit": self.control_limit,
            "windows": json.loads(self.table.to_json(orient="records")),
        }, indent=2)
