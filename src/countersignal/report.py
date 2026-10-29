"""Plain-language monthly report card for a clinical lead."""

from __future__ import annotations

import numpy as np
import pandas as pd

from .monitor import AWAITING, DRIFT, INSUFFICIENT, NO_UNTREATED, STABLE, MonitorResult
from .shift import psi_label


def _pct(x: float) -> str:
    return f"{x:.0%}"


def window_summary(row: pd.Series, result: MonitorResult) -> dict:
    band = result.config.oe_band
    lines = []
    status = row.get("oe_verdict", AWAITING)
    if status == NO_UNTREATED:
        headline = "No untreated patients"
        lines.append(f"All {int(row['n'])} patients were treated, so the untreated outcome cannot "
                     "be estimated this month (positivity failure).")
    elif status == AWAITING:
        headline = "Waiting for outcomes"
        lines.append(f"{int(row['n'])} patients scored; their outcomes are not recorded yet.")
    else:
        oe, lo, hi = row["oe_ipw"], row["oe_ipw_lo"], row["oe_ipw_hi"]
        direction = "more" if oe > 1 else "fewer"
        if status == STABLE:
            headline = "Calibration holding"
        elif status == DRIFT:
            headline = "Calibration has drifted"
        else:
            headline = "Not enough evidence yet"
        lines.append(
            f"Corrected for treatment, {abs(oe - 1):.0%} {direction} events than the model "
            f"predicted (observed/expected {oe:.2f}, 95% interval {lo:.2f} to {hi:.2f}; "
            f"acceptable range {band[0]:.2f} to {band[1]:.2f}).")
        needed = row.get("oe_window_needed_patients", np.nan)
        if status == INSUFFICIENT and not pd.isna(needed):
            lines.append(f"If the estimate holds, a month of about {int(needed):,} patients (this "
                         f"month had {int(row['n']):,}) would make the interval reach the edge of "
                         "the acceptable range, which settles the question about half the time.")
        naive = row.get("naive_oe_verdict")
        if naive is not None and naive != status:
            lines.append(f"A monitor that ignored treatment would have said: {naive} "
                         f"(observed/expected {row['oe_naive']:.2f}).")
        if row.get("auc_verdict") == DRIFT:
            lines.append(f"Ranking has weakened: corrected AUC {row['auc_ipw']:.3f} against "
                         f"{result.ref_auc:.3f} at validation.")
        if bool(row.get("cusum_alarm", False)):
            lines.append("The cumulative alarm is on: evidence of miscalibration has built up "
                         "across months beyond the planned false alarm rate.")
        if row.get("weights_truncated", 0) > 0:
            lines.append(f"{int(row['weights_truncated'])} untreated patients had a very low "
                         "chance of being untreated; their weights were capped, so read this "
                         "month with care.")
    lines.append(f"Treated after alerts or otherwise: {int(row['n_treated'])} of {int(row['n'])} "
                 f"({_pct(row['n_treated'] / row['n'])}).")
    lines.append(f"Case mix vs validation: {psi_label(row['score_psi'])} in risk scores "
                 f"(PSI {row['score_psi']:.2f}).")
    return {"window": row["window"], "headline": headline, "status": status, "lines": lines}


def report_card(result: MonitorResult) -> list[dict]:
    return [window_summary(row, result) for _, row in result.table.iterrows()]


def to_markdown(result: MonitorResult) -> str:
    out = ["# CounterSignal report card", ""]
    for card in report_card(result):
        out.append(f"## {card['window']}: {card['headline']}")
        out.extend(f"- {line}" for line in card["lines"])
        out.append("")
    return "\n".join(out)
