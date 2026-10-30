"""Reusable presentational components. No data fetching here."""
from __future__ import annotations

from html import escape
from pathlib import Path

import streamlit as st

from utils.formatting import or_missing, pct

_CSS = Path(__file__).resolve().parent.parent / "assets" / "styles.css"
_SEV_MARK = {
    "High": "\u25B2", "Medium": "\u25C6", "Low": "\u25CF",
    "Critical": "\u25B2", "Severe": "\u25B2", "Serious": "\u25C6", "Moderate": "\u25C6", "Minor": "\u25CF"
}  # shape + text, not color alone


def inject_css() -> None:
    st.markdown(f"<style>{_CSS.read_text(encoding='utf-8')}</style>", unsafe_allow_html=True)


def page_header(title: str, subtitle: str | None = None) -> None:
    st.title(title)
    if subtitle:
        st.markdown(f"<div class='pg-sub'>{escape(subtitle)}</div>", unsafe_allow_html=True)


def section(title: str, question: str | None = None) -> None:
    q = f"<p>{escape(question)}</p>" if question else ""
    st.markdown(f"<div class='sec'><h3>{escape(title)}</h3>{q}</div>", unsafe_allow_html=True)


def metric_card(label: str, value: str, note: str | None = None, html_value: bool = False) -> None:
    v = value if html_value else escape(str(value))
    n = f"<div class='note'>{escape(note)}</div>" if note else ""
    st.markdown(f"<div class='card'><div class='lab'>{escape(label)}</div><div class='val'>{v}</div>{n}</div>",
                unsafe_allow_html=True)


def severity_badge(label: str | None) -> str:
    lab = label if label in _SEV_MARK else "Unknown"
    return f"<span class='badge sev-{lab}'>{_SEV_MARK.get(lab, '?')} {escape(lab)}</span>"


def chips(items: list[str], factor: bool = False, empty: str = "None identified in the report") -> str:
    if not items:
        return f"<span class='none'>{escape(empty)}</span>"
    cls = "chip chip-f" if factor else "chip"
    return "".join(f"<span class='{cls}'>{escape(str(i))}</span>" for i in items)


def labeled_chips(label: str, items: list[str]) -> None:
    st.markdown(f"<div class='meta' style='margin-bottom:.3rem'>{escape(label)}</div>{chips(items)}", unsafe_allow_html=True)


def empty_state(title: str, body: str, page: str | None = None, page_label: str | None = None) -> None:
    st.markdown(f"<div class='empty'><b>{escape(title)}</b>{escape(body)}</div>", unsafe_allow_html=True)
    if page:
        st.page_link(page, label=page_label or "Continue")


def root_cause_card(rank: int, rc: dict) -> None:
    conf = rc.get("confidence")
    width = f"{max(0, min(1, float(conf))) * 100:.0f}%" if isinstance(conf, (int, float)) else "0%"
    ev = rc.get("evidence")
    evidence = (f"<blockquote>{escape(ev)}</blockquote>" if ev
                else "<p class='none'>The model did not return evidence text for this cause.</p>")
    st.markdown(
        f"<div class='rc'><div class='rc-top'><span class='rc-rank'>#{rank}</span>"
        f"<span class='rc-name'>{escape(rc.get('cause', 'Unnamed cause'))}</span>"
        f"<span class='rc-conf'>{pct(conf)}</span></div>"
        f"<div class='bar'><div style='width:{width}'></div></div><div class='meta'>Model confidence</div>"
        f"<div class='meta' style='margin-top:.7rem'>Supporting evidence from the report</div>{evidence}"
        f"<div class='meta' style='margin-bottom:.3rem'>Supporting factors</div>{chips(rc.get('supporting_factors', []), True, 'None returned')}</div>",
        unsafe_allow_html=True)


def severity_panel(sev: dict) -> None:
    score = sev.get("score")
    width = f"{max(0, min(1, float(score))) * 100:.0f}%" if isinstance(score, (int, float)) else "0%"
    st.markdown(
        f"<div class='card'>{severity_badge(sev.get('label'))}"
        f"<div class='bar sev'><div style='width:{width}'></div></div>"
        f"<div class='meta'>Severity score {escape(pct(score))} (model output, not a measured quantity)</div>"
        f"<div class='meta' style='margin:.7rem 0 .3rem'>Supporting factors</div>{chips(sev.get('supporting_factors', []), True, 'None returned')}</div>",
        unsafe_allow_html=True)


def ai_note() -> None:
    st.markdown("<div class='note-box'>Results are model-generated decision support. Root causes are probable, "
                "not confirmed, and confidence is the model's own estimate. Review them against the full investigation.</div>",
                unsafe_allow_html=True)


def demo_notice() -> None:
    from services import api_client
    if api_client.IS_DEMO_MODE:
        st.warning("DEMO MODE ACTIVE: Incident data shown here is synthetic mock data for demonstration only.")


def login_view() -> None:
    _, mid, _ = st.columns([1, 1.2, 1])
    with mid:
        st.title("Safety Incident Intelligence")
        st.markdown("<div class='pg-sub'>Analyze petrochemical incident reports, find probable root causes, and compare with history.</div>", unsafe_allow_html=True)
        with st.form("login"):
            name = st.text_input("Your name")
            role = st.selectbox("Role", ["Normal user", "Safety analyst / admin"])
            if st.form_submit_button("Sign in", type="primary", width="stretch"):
                if not name.strip():
                    st.error("Enter your name to continue.")
                else:
                    st.session_state["user"] = {"name": name.strip(), "role": "analyst" if role.startswith("Safety") else "user"}
                    st.rerun()
        st.caption("Demo sign-in placeholder. Real authentication will be handled by the backend API.")


def factor_columns(ex: dict) -> None:
    cols = st.columns(4)
    for col, (label, key) in zip(cols, [("Human", "human_factors"), ("Maintenance", "maintenance_factors"),
                                        ("Operational", "operational_factors"), ("Environmental", "environmental_factors")]):
        with col:
            labeled_chips(label, ex.get(key, []))


def historical_tables(hist: dict) -> None:
    import pandas as pd
    t1, t2, t3 = st.tabs(["Similar incidents", "Same equipment", "Recurring root causes"])
    cfg = {"id": "Incident ID", "date": "Date", "type": "Type", "equipment": "Equipment", "severity": "Severity",
           "root_cause": "Root cause", "similarity": "Match score", "cause": "Root cause", "total": "Incidents in history",
           "on_same_equipment": "On this equipment", "last_seen": "Last seen"}
    def show(rows, empty):
        if rows:
            st.dataframe(pd.DataFrame(rows), hide_index=True, width="stretch", column_config={k: v for k, v in cfg.items()})
        else:
            st.info(empty)
    with t1:
        show(hist.get("similar_incidents"), "No sufficiently similar past incidents were found.")
    with t2:
        show(hist.get("same_equipment_incidents"), "No equipment was identified in the report, or no past incidents involve it.")
    with t3:
        show(hist.get("recurring_root_causes"), "Recurring root-cause information is unavailable for this analysis.")
