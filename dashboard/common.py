"""Shared helpers for the CounterSignal dashboard."""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
RESULTS = Path(os.environ.get("COUNTERSIGNAL_RESULTS", ROOT / "results"))

TRUTH = "#111827"
NAIVE = "#6b7280"
UNTREATED = "#b45309"
COUNTER = "#0f766e"
METHOD_LABELS = {"naive": "Standard monitor (all patients)",
                 "untreated_only": "Untreated patients only",
                 "countersignal": "CounterSignal"}


def page_setup(title: str) -> None:
    st.set_page_config(page_title=f"{title} · CounterSignal", page_icon="📡", layout="wide")


@st.cache_data
def summary() -> dict | None:
    path = RESULTS / "benchmark_summary.json"
    return json.loads(path.read_text()) if path.exists() else None


@st.cache_data
def example(scenario: str) -> pd.DataFrame | None:
    path = RESULTS / f"example_{scenario}.csv"
    return pd.read_csv(path) if path.exists() else None


def footnote(text: str) -> None:
    st.caption(text)


def oe_chart(table: pd.DataFrame, band=(0.8, 1.25), truth: bool = True) -> go.Figure:
    x = table["window"]
    fig = go.Figure()
    fig.add_hrect(y0=band[0], y1=band[1], fillcolor="#e5e7eb", opacity=0.45, line_width=0,
                  annotation_text="acceptable range", annotation_position="top left")
    fig.add_trace(go.Scatter(x=list(x) + list(x[::-1]),
                             y=list(table["oe_ipw_hi"]) + list(table["oe_ipw_lo"][::-1]),
                             fill="toself", fillcolor="rgba(15,118,110,0.15)", line_width=0,
                             name="CounterSignal 95% interval", hoverinfo="skip"))
    if truth and "true_oe" in table:
        fig.add_trace(go.Scatter(x=x, y=table["true_oe"], name="True (known in benchmark)",
                                 line=dict(color=TRUTH, width=3, dash="dot")))
    fig.add_trace(go.Scatter(x=x, y=table["oe_naive"], name=METHOD_LABELS["naive"],
                             line=dict(color=NAIVE, width=2)))
    fig.add_trace(go.Scatter(x=x, y=table["oe_untreated"], name=METHOD_LABELS["untreated_only"],
                             line=dict(color=UNTREATED, width=2)))
    fig.add_trace(go.Scatter(x=x, y=table["oe_ipw"], name=METHOD_LABELS["countersignal"],
                             line=dict(color=COUNTER, width=3)))
    fig.update_layout(height=420, margin=dict(l=10, r=10, t=10, b=10),
                      yaxis_title="Observed / expected events", xaxis_title="Month",
                      legend=dict(orientation="h", y=-0.22), hovermode="x unified")
    return fig


def cusum_chart(table: pd.DataFrame, limit: float | None) -> go.Figure:
    fig = go.Figure()
    for method, suffix, color in (("naive", "_naive", NAIVE), ("untreated_only", "_untreated", UNTREATED),
                                  ("countersignal", "", COUNTER)):
        level = table[[f"cusum_upper{suffix}", f"cusum_lower{suffix}"]].max(axis=1)
        fig.add_trace(go.Scatter(x=table["window"], y=level, name=METHOD_LABELS[method],
                                 line=dict(color=color, width=3 if suffix == "" else 2)))
    if limit is not None:
        fig.add_hline(y=limit, line_dash="dash", line_color="#dc2626",
                      annotation_text="alarm limit", annotation_position="top left")
    fig.update_layout(height=330, margin=dict(l=10, r=10, t=10, b=10),
                      yaxis_title="Accumulated evidence of miscalibration", xaxis_title="Month",
                      legend=dict(orientation="h", y=-0.3), hovermode="x unified")
    return fig
