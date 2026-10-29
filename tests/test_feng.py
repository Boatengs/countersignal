import numpy as np

from countersignal.baselines.feng_score_cusum import FengScoreCUSUM


def _gen(rng, n, shift=0.0):
    s = rng.normal(-2.2, 0.8, n)
    c = rng.normal(size=n)
    y = (rng.random(n) < 1 / (1 + np.exp(-(s + 0.3 * c + shift)))).astype(float)
    return np.column_stack([s, c]), y


def test_feng_false_alarm_rate_and_power():
    alarms = detected = 0
    reps = 60
    for r in range(reps):
        rng = np.random.default_rng(r)
        m = FengScoreCUSUM(horizon=8, seed=r)
        m.start(*_gen(rng, 6000))
        for _ in range(8):
            m.update(*_gen(rng, 800))
        alarms += m.alarmed
        m = FengScoreCUSUM(horizon=8, seed=r)
        m.start(*_gen(rng, 6000))
        for t in range(8):
            m.update(*_gen(rng, 800, 0.4 if t >= 3 else 0.0))
        detected += m.alarmed
    assert alarms / reps <= 0.12  # nominal 5%, allowing for 60 replicates
    assert detected / reps >= 0.9
