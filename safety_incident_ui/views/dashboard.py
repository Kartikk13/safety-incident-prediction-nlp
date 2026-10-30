import streamlit as st

from components import charts, ui
from services import api_client

ui.page_header(
    "Dashboard",
    "Real-time safety incident KPIs and distribution patterns aggregated directly from PostgreSQL.",
)
ui.demo_notice()

# Population Selector
pop_choice = st.radio(
    "Data population",
    ["All records", "User submissions only", "Historical dataset only"],
    horizontal=True,
    key="dash_pop_filter",
)
source_param = None
if pop_choice == "User submissions only":
    source_param = "user_submission"
elif pop_choice == "Historical dataset only":
    source_param = "historical_dataset"

try:
    with st.spinner("Loading dashboard KPIs from PostgreSQL..."):
        kpis = api_client.get_kpis(source=source_param)
except api_client.ApiError as e:
    st.error(str(e))
    st.stop()

total_incidents = kpis.get("total_incidents", 0)
if total_incidents == 0:
    ui.empty_state(
        "No incidents in selected population",
        "Select a different population or submit a new incident report.",
        "views/new_incident.py",
        "New incident",
    )
    st.stop()

# Metric cards
c = st.columns(4)
with c[0]:
    ui.metric_card(
        "Incidents on record",
        f"{total_incidents:,}",
        f"{kpis.get('historical_count', 0):,} historical · {kpis.get('user_submission_count', 0):,} user submitted",
    )
with c[1]:
    high_count = kpis.get("high_severity_count", 0)
    high_pct = kpis.get("high_severity_pct", 0.0)
    ui.metric_card(
        "High / Critical severity",
        f"{high_count:,}",
        f"{high_pct:.1f}% of population ({high_count}/{total_incidents})",
    )
with c[2]:
    l90 = kpis.get("last_90_days_count", 0)
    delta = kpis.get("last_90_days_delta", 0)
    ui.metric_card(
        "Last 90 days",
        f"{l90:,}",
        f"{delta:+d} vs previous 90 days",
    )
with c[3]:
    top_eq = kpis.get("top_high_severity_equipment_past_year", "None")
    eq_cnt = kpis.get("top_equipment_high_severity_count", 0)
    ui.metric_card(
        "Most high-severity asset (past year)",
        top_eq,
        f"{eq_cnt} incidents" if eq_cnt else "No incidents in past year",
    )

# Trends Chart
ui.section(
    "Incident timeline trend",
    f"Monthly incident counts for {pop_choice.lower()} aggregated in PostgreSQL.",
)
try:
    trends = api_client.get_trends(source=source_param)
    charts.show(charts.trend_line_chart(trends))
except api_client.ApiError as e:
    st.warning(f"Could not load trends: {e}")

# Categorical Distributions
a, b = st.columns(2)
with a:
    ui.section(
        "Leading root causes",
        "Top recurring root cause categories aggregated across records.",
    )
    try:
        cause_dim = "predicted_root_cause" if source_param == "user_submission" else "root_cause"
        rc_dist = api_client.get_distributions(dim=cause_dim, source=source_param, limit=8)
        charts.show(charts.distribution_bar_chart(rc_dist, label="Root cause", top=8))
    except api_client.ApiError as e:
        st.warning(f"Could not load root causes: {e}")

with b:
    ui.section(
        "Equipment patterns",
        "Equipment assets with the highest incident frequencies.",
    )
    try:
        eq_dist = api_client.get_distributions(dim="equipment_type", source=source_param, limit=8)
        charts.show(charts.distribution_bar_chart(eq_dist, label="Equipment type", top=8, color="#2F5D8A"))
    except api_client.ApiError as e:
        st.warning(f"Could not load equipment patterns: {e}")

# Secondary row
a, b = st.columns([1, 1.3])
with a:
    ui.section(
        "Incident types",
        "Distribution of incident classifications.",
    )
    try:
        type_dist = api_client.get_distributions(dim="incident_type", source=source_param, limit=6)
        charts.show(charts.distribution_bar_chart(type_dist, label="Incident type", top=6, color="#C77D0A"))
    except api_client.ApiError as e:
        st.warning(f"Could not load incident types: {e}")

with b:
    ui.section("Recent incident records")
    try:
        df_recent = api_client.list_incidents(source=source_param, limit=8)
        if not df_recent.empty:
            st.dataframe(
                df_recent.head(8)[["id", "date", "type", "equipment", "severity"]],
                hide_index=True,
                width="stretch",
                column_config={
                    "id": "Incident ID",
                    "date": st.column_config.DateColumn("Date", format="YYYY-MM-DD"),
                    "type": "Type",
                    "equipment": "Equipment",
                    "severity": "Severity",
                },
            )
        else:
            st.info("No recent incidents.")
    except api_client.ApiError as e:
        st.warning(f"Could not load recent incidents: {e}")
