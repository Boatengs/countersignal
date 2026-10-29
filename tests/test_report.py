import numpy as np

from countersignal import Monitor, MonitorConfig
from countersignal.report import report_card, to_markdown
from countersignal.synthetic import make_deployment


def test_report_card_covers_every_window_and_flags_naive_disagreement():
    ref, log = make_deployment(n_per_window=4000, windows=3, seed=11)
    log.loc[log.window == "W03", "outcome"] = np.nan
    res = Monitor(ref, MonitorConfig(covariates=["x1", "x2", "alerted"], n_boot=60, horizon=3)).run(log)
    cards = report_card(res)
    assert [c["window"] for c in cards] == ["W01", "W02", "W03"]
    assert cards[2]["headline"] == "Waiting for outcomes"
    text = to_markdown(res)
    assert "ignored treatment" in text
    assert "nan" not in text.lower()
