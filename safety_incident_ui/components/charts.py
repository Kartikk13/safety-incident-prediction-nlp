"""Plotly charts with one shared theme. Each function takes a DataFrame of incidents."""
from __future__ import annotations

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

INK, ACCENT, GRID = "#14202B", "#0F766E", "#E3E8EC"
SEV_COLORS = {
    "Low": "#2F5D8A", "Medium": "#C77D0A", "High": "#B42318",
    "Minor": "#2F5D8A", "Moderate": "#3B82F6", "Serious": "#C77D0A", "Severe": "#B45309", "Critical": "#B42318"
}
SEV_ORDER = ["Minor", "Moderate", "Serious", "Severe", "Critical", "Low", "Medium", "High"]


def _style(fig: go.Figure, height: int = 320) -> go.Figure:
    fig.update_layout(template="simple_white", height=height, margin=dict(l=8, r=8, t=8, b=8),
                      font=dict(family="IBM Plex Sans, sans-serif", size=13, color=INK),
                      paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                      legend=dict(orientation="h", y=-0.22, title=None))
    fig.update_xaxes(showgrid=False)
    fig.update_yaxes(gridcolor=GRID)
    return fig


def show(fig: go.Figure) -> None:
    st.plotly_chart(fig, width="stretch", config={"displayModeBar": False})


def monthly_trend(df: pd.DataFrame) -> go.Figure:
    m = df.assign(month=df["date"].dt.to_period("M").dt.to_timestamp())
    total = m.groupby("month").size()
    high = m[m["severity"].isin(["High", "Critical", "Severe", "Serious"])].groupby("month").size().reindex(total.index, fill_value=0)
    fig = go.Figure()
    fig.add_scatter(x=total.index, y=total.values, name="All incidents", mode="lines+markers", line=dict(color=ACCENT, width=2.5))
    fig.add_scatter(x=high.index, y=high.values, name="High severity", mode="lines+markers", line=dict(color=SEV_COLORS["High"], width=2.5, dash="dot"))
    fig.update_yaxes(title="Incidents per month", rangemode="tozero")
    return _style(fig)


def category_bar(df: pd.DataFrame, col: str, label: str, top: int = 8) -> go.Figure:
    c = df[col].value_counts().head(top).sort_values().rename_axis(label).reset_index(name="Incidents")
    fig = px.bar(c, x="Incidents", y=label, orientation="h", color_discrete_sequence=[ACCENT])
    fig.update_yaxes(title=None, gridcolor="rgba(0,0,0,0)")
    fig.update_xaxes(title="Incidents", showgrid=True, gridcolor=GRID)
    return _style(fig, 300)


def severity_by_cause(df: pd.DataFrame) -> go.Figure:
    c = df.groupby(["root_cause", "severity"]).size().reset_index(name="Incidents")
    order = df["root_cause"].value_counts().index[::-1].tolist()
    fig = px.bar(c, x="Incidents", y="root_cause", color="severity", orientation="h", color_discrete_map=SEV_COLORS,
                 category_orders={"severity": SEV_ORDER, "root_cause": order}, labels={"root_cause": "", "severity": "Severity"})
    return _style(fig, 340)


def cause_heatmap(df: pd.DataFrame) -> go.Figure:
    ct = pd.crosstab(df["equipment"], df["root_cause"])
    fig = px.imshow(ct, aspect="auto", color_continuous_scale=["#F1F6F6", ACCENT], text_auto=True,
                    labels=dict(x="Root cause", y="Equipment", color="Incidents"))
    fig.update_xaxes(side="bottom", tickangle=-30)
    return _style(fig, 400)


def cause_trend_quarterly(df: pd.DataFrame, top: int = 4) -> go.Figure:
    causes = df["root_cause"].value_counts().head(top).index
    q = (df[df["root_cause"].isin(causes)].assign(quarter=df["date"].dt.to_period("Q").dt.to_timestamp())
         .groupby(["quarter", "root_cause"]).size().reset_index(name="Incidents"))
    fig = px.line(q, x="quarter", y="Incidents", color="root_cause", markers=True, labels={"quarter": "", "root_cause": ""},
                  color_discrete_sequence=["#0F766E", "#2F5D8A", "#B45309", "#6B7A8C"])
    fig.update_yaxes(rangemode="tozero")
    return _style(fig)


def equipment_timeline(df: pd.DataFrame) -> go.Figure:
    fig = px.scatter(df, x="date", y="severity", color="severity", color_discrete_map=SEV_COLORS,
                     category_orders={"severity": SEV_ORDER}, hover_data={"id": True, "root_cause": True, "type": True, "severity": False},
                     labels={"date": "", "severity": "Severity"})
    fig.update_traces(marker=dict(size=11, line=dict(width=1, color="white")))
    fig.update_layout(showlegend=False)
    return _style(fig, 260)


def trend_line_chart(trends: list[dict]) -> go.Figure:
    """Render a monthly trend line chart from SQL date-aggregated buckets."""
    if not trends:
        fig = go.Figure()
        return _style(fig)
    df = pd.DataFrame(trends)
    fig = go.Figure()
    fig.add_scatter(
        x=df["month"],
        y=df["total"],
        name="All incidents",
        mode="lines+markers",
        line=dict(color=ACCENT, width=2.5),
    )
    if "high_severity" in df.columns:
        fig.add_scatter(
            x=df["month"],
            y=df["high_severity"],
            name="High / Critical severity",
            mode="lines+markers",
            line=dict(color=SEV_COLORS["High"], width=2.5, dash="dot"),
        )
    fig.update_xaxes(title="Month (YYYY-MM)", tickangle=-30)
    fig.update_yaxes(title="Incidents per month", rangemode="tozero")
    return _style(fig)


def distribution_bar_chart(
    dist: list[dict],
    label: str = "Category",
    top: int = 10,
    color: str = ACCENT,
) -> go.Figure:
    """Render horizontal bar chart directly from SQL-aggregated category count items."""
    if not dist:
        fig = go.Figure()
        return _style(fig, 300)
    df = pd.DataFrame(dist[:top])
    # Reverse so largest is at the top of the horizontal bar chart
    df = df.iloc[::-1].reset_index(drop=True)
    fig = px.bar(
        df,
        x="count",
        y="name",
        orientation="h",
        color_discrete_sequence=[color],
        labels={"name": label, "count": "Incidents"},
    )
    fig.update_yaxes(title=None, gridcolor="rgba(0,0,0,0)")
    fig.update_xaxes(title="Incidents (Count)", showgrid=True, gridcolor=GRID)
    return _style(fig, max(260, 30 * len(df)))


def matrix_heatmap(heatmap_data: dict) -> go.Figure:
    """Render cross-tabulation heatmap from backend SQL matrix output."""
    equipment = heatmap_data.get("equipment", [])
    causes = heatmap_data.get("root_causes", [])
    matrix = heatmap_data.get("matrix", [])
    if not equipment or not causes or not matrix:
        fig = go.Figure()
        return _style(fig, 300)
    fig = px.imshow(
        matrix,
        x=causes,
        y=equipment,
        aspect="auto",
        color_continuous_scale=["#F1F6F6", ACCENT],
        text_auto=True,
        labels=dict(x="Root cause category", y="Equipment asset", color="Incidents"),
    )
    fig.update_xaxes(side="bottom", tickangle=-30)
    return _style(fig, max(320, 28 * len(equipment)))

