"""The CounterSignal monitor: one row of evidence per monitoring window."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from . import metrics as M
from .sequential import control_limit, cusum_path, exact_z
from .shift import psi
from .weights import design, effective_sample_size, fit_propensity, ipw_weights

STABLE = "stable"
DRIFT = "drift"
INSUFFICIENT = "not enough evidence yet"
AWAITING = "awaiting outcomes"


@dataclass
class MonitorConfig:
    score: str = "score"
    outcome: str = "outcome"
    treated: str = "treated"
    window: str = "window"
    covariates: list[str] = field(default_factory=list)
    oe_band: tuple[float, float] = (0.8, 1.25)
    auc_tolerance: float = 0.05
    alpha: float = 0.05
    horizon: int = 24
    cusum_k: float = 0.5
    n_boot: int = 200
    weight_floor: float = 0.02
    min_labelled: int = 1
    seed: int = 0


def _band_verdict(lo: float, hi: float, band: tuple[float, float]) -> str:
    if np.isnan(lo) or np.isnan(hi):
        return INSUFFICIENT
    if hi < band[0] or lo > band[1]:
        return DRIFT
    if lo >= band[0] and hi <= band[1]:
        return STABLE
    return INSUFFICIENT


def _oe_cases_needed(oe: float, lo: float, hi: float, n_eff: float,
                     band: tuple[float, float]) -> float:
    """Approximate effective sample size at which the O/E interval would settle.

    Assumes the interval width shrinks with the square root of the effective
    sample size and the point estimate stays where it is. Returns NaN when the
    estimate sits on a band edge, where no finite sample settles it.
    """
    if any(np.isnan(v) for v in (oe, lo, hi)) or oe <= 0 or lo <= 0:
        return float("nan")
    half = (np.log(hi) - np.log(lo)) / 2
    distance = min(abs(np.log(oe) - np.log(band[0])), abs(np.log(oe) - np.log(band[1])))
    if distance < 1e-3:
        return float("nan")
    return float(np.ceil(n_eff * (half / distance) ** 2))


class Monitor:
    """Monitor a deployed clinical risk model for drift, correcting for the
    interventions its own alerts trigger.

    reference: pre-deployment validation data with the model's score and the
    outcome, collected while the model did not influence care.
    """

    def __init__(self, reference: pd.DataFrame, config: MonitorConfig | None = None):
        self.cfg = config or MonitorConfig()
        c = self.cfg
        ref = reference.dropna(subset=[c.score, c.outcome])
        self.reference = ref
        y, p = ref[c.outcome].to_numpy(float), ref[c.score].to_numpy(float)
        self.ref_auc = M.auc(y, p)
        self.ref_oe = M.observed_expected(y, p)
        self.limit = control_limit(c.horizon, c.alpha, c.cusum_k)  # fixed seed: same limit every run
        self.outcome_model = None
        if c.covariates and len(np.unique(y)) == 2:
            self.outcome_model = make_pipeline(
                StandardScaler(), LogisticRegression(C=1e4, max_iter=5000))
            self.outcome_model.fit(design(ref, c.score, c.covariates), y.astype(int))

    def _m0(self, frame: pd.DataFrame) -> np.ndarray:
        if self.outcome_model is None:
            return frame[self.cfg.score].to_numpy(float)
        return self.outcome_model.predict_proba(design(frame, self.cfg.score, self.cfg.covariates))[:, 1]

    def run(self, log: pd.DataFrame) -> MonitorResult:
        c = self.cfg
        rng = np.random.default_rng(c.seed)
        windows = list(pd.unique(log[c.window]))
        rows = []
        for i, win in enumerate(windows):
            seen = log[log[c.window].isin(windows[: i + 1])]
            prop = fit_propensity(seen, c.score, c.treated, c.covariates)
            cur = log[log[c.window] == win]
            rows.append(self._evaluate(win, cur, prop, rng))
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

    def _evaluate(self, win, cur: pd.DataFrame, prop, rng) -> dict:
        c = self.cfg
        out: dict = {"window": win, "n": len(cur), "n_treated": int(cur[c.treated].sum())}
        out["mean_score"] = float(cur[c.score].mean())
        out["score_psi"] = psi(self.reference[c.score], cur[c.score])
        cov_psi = {col: psi(self.reference[col], cur[col]) for col in c.covariates}
        if cov_psi:
            top = max(cov_psi, key=lambda k: cov_psi[k])
            out["top_shift_feature"], out["top_shift_psi"] = top, cov_psi[top]
        lab = cur.dropna(subset=[c.outcome])
        out["n_labelled"] = len(lab)
        if len(lab) < c.min_labelled:
            out.update({"oe_verdict": AWAITING, "auc_verdict": AWAITING,
                        "naive_oe_verdict": AWAITING, "untreated_oe_verdict": AWAITING,
                        "z": np.nan, "z_naive": np.nan, "z_untreated": np.nan})
            return out
        y = lab[c.outcome].to_numpy(float)
        p = lab[c.score].to_numpy(float)
        a = lab[c.treated].to_numpy(int)
        p0 = prop.p_untreated(lab, c.score)
        w = ipw_weights(a, p0, c.weight_floor)
        m0 = self._m0(lab)
        untreated = (1 - a).astype(float)
        out["ess"] = effective_sample_size(w)
        out["weights_truncated"] = int(np.sum((a == 0) & (p0 < c.weight_floor)))

        def stats(idx):
            yy, pp, ww, mm, uu = y[idx], p[idx], w[idx], m0[idx], untreated[idx]
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

        point = stats(np.arange(len(y)))
        out.update(point)
        out["cal_intercept_ipw"], out["cal_slope_ipw"] = M.calibration_intercept_slope(y, p, w)
        boots = [stats(rng.integers(0, len(y), len(y))) for _ in range(c.n_boot)]
        frame = pd.DataFrame(boots)
        for key in ("oe_naive", "auc_naive", "oe_untreated", "auc_untreated",
                    "oe_ipw", "auc_ipw", "oe_dr"):
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
        out["oe_cases_needed"] = (
            _oe_cases_needed(out["oe_ipw"], out["oe_ipw_lo"], out["oe_ipw_hi"], out["ess"], c.oe_band)
            if out["oe_verdict"] == INSUFFICIENT else np.nan)
        out["z"] = exact_z(y, p, w, rng)
        out["z_naive"] = exact_z(y, p, np.ones_like(y), rng)
        out["z_untreated"] = exact_z(y, p, untreated, rng)
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
