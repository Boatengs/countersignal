import numpy as np
import pytest

from countersignal import DRIFT, INSUFFICIENT, STABLE, Monitor, MonitorConfig
from countersignal import metrics as M
from countersignal.sequential import control_limit, cusum_path, exact_z
from countersignal.synthetic import make_deployment

CFG = dict(covariates=["x1", "x2", "alerted"], n_boot=100, horizon=12)


def test_ipw_recovers_untreated_performance_when_naive_does_not():
    ref, log = make_deployment(n_per_window=20_000, windows=1, seed=3)
    true_oe = M.observed_expected(log.y0, log.score)
    res = Monitor(ref, MonitorConfig(**CFG)).run(log)
    row = res.table.iloc[0]
    assert row.oe_naive < 0.9  # prevented events make the model look like it over-predicts
    assert row.oe_ipw == pytest.approx(true_oe, abs=0.04)
    assert row.oe_dr == pytest.approx(true_oe, abs=0.04)
    assert row.auc_ipw == pytest.approx(M.auc(log.y0, log.score), abs=0.01)
    assert row.auc_naive < M.auc(log.y0, log.score) - 0.01
    assert row.oe_verdict == STABLE
    assert row.naive_oe_verdict != STABLE


def test_untreated_only_misses_drift_in_the_patients_who_get_treated():
    ref, log = make_deployment(n_per_window=40_000, windows=1, seed=4, odds_shift_from=0,
                               odds_ratio=2.0, shift_high_risk_only=True)
    true_oe = M.observed_expected(log.y0, log.score)
    true_auc = M.auc(log.y0, log.score)
    row = Monitor(ref, MonitorConfig(**CFG)).run(log).table.iloc[0]
    assert abs(row.oe_untreated - true_oe) > abs(row.oe_ipw - true_oe) + 0.03
    assert abs(row.auc_untreated - true_auc) > abs(row.auc_ipw - true_auc) + 0.03
    assert row.oe_ipw == pytest.approx(true_oe, abs=0.04)


def test_real_drift_is_detected():
    ref, log = make_deployment(n_per_window=3000, windows=12, seed=5,
                               odds_shift_from=4, odds_ratio=1.8)
    res = Monitor(ref, MonitorConfig(**CFG)).run(log)
    late = res.table.iloc[6:]
    assert (late.oe_verdict == DRIFT).sum() >= 4
    assert (late.oe_verdict != STABLE).all()
    assert (res.table.iloc[:4].oe_verdict != DRIFT).all()
    assert res.first_alarm is not None
    assert int(res.first_alarm[1:]) >= 5  # no alarm before the shift begins


def test_small_windows_give_honest_uncertainty():
    ref, log = make_deployment(n_per_window=120, windows=6, seed=6)
    res = Monitor(ref, MonitorConfig(**CFG)).run(log)
    assert (res.table.oe_verdict == INSUFFICIENT).sum() >= 4
    assert res.table.oe_cases_needed.notna().any()


def test_exact_z_is_standard_normal_under_null_even_when_small():
    rng = np.random.default_rng(7)
    zs = []
    for _ in range(600):
        p = rng.uniform(0.02, 0.3, 40)
        y = (rng.random(40) < p).astype(int)
        w = rng.uniform(1, 3, 40)
        zs.append(exact_z(y, p, w, rng, n_sim=1000))
    zs = np.array(zs)
    assert abs(zs.mean()) < 0.12
    assert abs(zs.std() - 1) < 0.12


def test_control_limit_false_alarm_rate():
    h = control_limit(horizon=12, alpha=0.05, seed=1)
    rng = np.random.default_rng(8)
    alarms = 0
    for _ in range(4000):
        path = cusum_path(list(rng.standard_normal(12)))
        alarms += max(max(s.upper, s.lower) for s in path) > h
    assert alarms / 4000 == pytest.approx(0.05, abs=0.012)


def test_awaiting_outcomes_window():
    ref, log = make_deployment(n_per_window=500, windows=3, seed=9)
    log.loc[log.window == "W03", "outcome"] = np.nan
    t = Monitor(ref, MonitorConfig(**CFG)).run(log).table
    assert t.iloc[2].oe_verdict == "awaiting outcomes"
    assert not np.isnan(t.iloc[2].score_psi)
