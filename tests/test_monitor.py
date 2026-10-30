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
    assert row.oe_naive_hi < true_oe  # the naive interval excludes the truth
    assert row.oe_ipw_lo <= true_oe <= row.oe_ipw_hi


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
    assert res.table.oe_window_needed_patients.notna().any()
    need = res.table.dropna(subset=["oe_window_needed_patients"])
    assert (need.oe_window_needed_patients > need.n).all()  # a bigger window than this month


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


def test_windows_are_processed_in_time_order_whatever_the_row_order():
    ref, log = make_deployment(n_per_window=400, windows=4, seed=12)
    shuffled = log.sample(frac=1.0, random_state=0)
    a = Monitor(ref, MonitorConfig(**CFG)).run(log).table
    b = Monitor(ref, MonitorConfig(**CFG)).run(shuffled).table
    assert list(b.window) == ["W01", "W02", "W03", "W04"]
    assert np.allclose(a.oe_ipw, b.oe_ipw)


def test_all_treated_window_reports_positivity_failure():
    from countersignal.monitor import NO_UNTREATED
    from countersignal.report import to_markdown
    ref, log = make_deployment(n_per_window=300, windows=2, seed=13)
    log.loc[log.window == "W02", "treated"] = 1
    res = Monitor(ref, MonitorConfig(**CFG)).run(log)
    assert res.table.iloc[1].oe_verdict == NO_UNTREATED
    assert "nan" not in to_markdown(res).lower()


def test_partly_recorded_window_waits_for_outcomes():
    ref, log = make_deployment(n_per_window=300, windows=2, seed=14)
    idx = log.index[log.window == "W02"][:50]
    log.loc[idx, "outcome"] = np.nan
    assert Monitor(ref, MonitorConfig(**CFG)).run(log).table.iloc[1].oe_verdict == "awaiting outcomes"


def test_invalid_inputs_fail_clearly():
    ref, log = make_deployment(n_per_window=200, windows=1, seed=15)
    bad = log.copy()
    bad.loc[bad.index[0], "treated"] = np.nan
    with pytest.raises(ValueError, match="missing values"):
        Monitor(ref, MonitorConfig(**CFG)).run(bad)
    with pytest.raises(ValueError, match="n_boot"):
        Monitor(ref, MonitorConfig(**{**CFG, "n_boot": 5}))


def test_wrong_column_roles_fail_clearly():
    ref, log = make_deployment(n_per_window=200, windows=1, seed=16)
    # a raw covariate is not a predicted risk
    with pytest.raises(ValueError, match="between 0 and 1"):
        Monitor(ref, MonitorConfig(**{**CFG, "score": "x1", "covariates": ["x2", "alerted"]}))
    # a continuous column is not a treatment flag
    with pytest.raises(ValueError, match="0/1"):
        Monitor(ref, MonitorConfig(**{**CFG, "covariates": ["x2", "alerted"],
                                      "treated": "x1"})).run(log)
    # the same column cannot fill two roles
    with pytest.raises(ValueError, match="distinct"):
        Monitor(ref, MonitorConfig(**{**CFG, "treated": "alerted"})).run(log)
    # covariates must be numeric
    text = log.assign(site="A")
    with pytest.raises(ValueError, match="numeric"):
        Monitor(ref.assign(site="A"), MonitorConfig(**{**CFG, "covariates": ["x1", "site"]})).run(text)


def test_end_to_end_false_alarm_rate_of_monitor():
    """Whole pipeline (estimated weights, exact null, CUSUM) under no change: planned 5%."""
    alarms = 0
    reps = 40
    for r in range(reps):
        ref, log = make_deployment(n_per_window=800, windows=6, seed=1000 + r)
        cfg = MonitorConfig(covariates=["x1", "x2", "alerted"], n_boot=20, horizon=6, seed=r,
                            bootstrap_refit=False)
        alarms += Monitor(ref, cfg).run(log).first_alarm is not None
    assert alarms / reps <= 0.15
