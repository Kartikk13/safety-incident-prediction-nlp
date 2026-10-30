import streamlit as st

from components import ui
from services import api_client
from utils.formatting import or_missing, pct

res = st.session_state.get("analysis")
if not res:
    # Attempt to retrieve saved incident analysis using query param or session state ID
    target_id = st.query_params.get("id") or st.session_state.get("current_incident_id")
    if target_id:
        try:
            with st.spinner(f"Loading analysis for incident {target_id}..."):
                res = api_client.get_incident_analysis(target_id)
                st.session_state["analysis"] = res
                st.session_state["current_incident_id"] = target_id
        except api_client.ApiError as e:
            st.error(str(e))
            st.stop()

if not res:
    ui.page_header("Incident analysis")
    ui.empty_state("No analysis to show yet", "Provide an incident report and run an analysis to see results here.", "views/new_incident.py", "Start a new analysis")
    st.stop()

inc, ex, sev, hist = res["incident"], res["extracted_information"], res["severity"], res["historical_analysis"]
causes = res["root_causes"]
ui.page_header(f"Incident {inc.get('id', '')}", f"{or_missing(inc.get('type'), 'Type not determined')}. Analysis complete.")
ui.ai_note()

a, b, c = st.columns(3)
with a: ui.metric_card("Incident type", or_missing(inc.get("type"), "Not determined"))
with b: ui.metric_card("Severity classification", ui.severity_badge(sev.get("label")), f"Score {pct(sev.get('score'))}", html_value=True)
with c: ui.metric_card("Leading probable cause", causes[0]["cause"] if causes else "Not available", f"Model confidence {pct(causes[0].get('confidence'))}" if causes else "No root cause was returned")

ui.section("Incident overview", "What happened, when, and where?")
st.write(or_missing(inc.get("summary"), "No summary was produced."))
d, t, l = st.columns(3)
d.markdown(f"**Date**  \n{or_missing(inc.get('date'))}"); t.markdown(f"**Time**  \n{or_missing(inc.get('time'))}"); l.markdown(f"**Location**  \n{or_missing(inc.get('location'))}")

ui.section("Assets and hazards", "What was involved?")
x, y, z = st.columns(3)
with x: ui.labeled_chips("Equipment", ex["equipment"])
with y: ui.labeled_chips("Chemicals and materials", ex["materials"])
with z: ui.labeled_chips("Hazards", ex["hazards"])

ui.section("Contributing factors", "What circumstances contributed?")
ui.factor_columns(ex)

ui.section("Probable root causes", "What are the likely causes, how confident is the model, and what supports them?")
if causes:
    for i, rc in enumerate(causes, 1):
        ui.root_cause_card(i, rc)
else:
    st.info("Root-cause results are unavailable for this report.")

ui.section("Severity", "How severe is the incident?")
ui.severity_panel(sev)

ui.section("Historical comparison", "Has something similar happened before, and is there a pattern?")
ui.historical_tables(hist)
with st.expander("Source report"):
    st.text(res.get("report_text") or "Source text is not available for this incident.")
