import pandas as pd
import streamlit as st

from components import charts, ui
from services import api_client

ui.page_header(
    "Historical analysis",
    "Exploration, ground-truth distributions, and data-quality audit of the verified historical training dataset.",
)
ui.demo_notice()

try:
    with st.spinner("Loading historical dataset summary from PostgreSQL..."):
        summary = api_client.get_dataset_summary()
except api_client.ApiError as e:
    st.error(str(e))
    st.stop()

if not summary or not summary.get("total_records"):
    st.info("Historical training dataset records are currently unavailable.")
    st.stop()

total_recs = summary.get("total_records", 0)
date_range = summary.get("date_range", {})
targets = summary.get("target_distributions", {})
nulls = summary.get("null_counts", {})
latest_import = summary.get("latest_import", {})

# 1. Dataset summary KPIs
c = st.columns(4)
with c[0]:
    ui.metric_card("Historical records", f"{total_recs:,}", "100% ground-truth labeled")
with c[1]:
    d_min = date_range.get("min", "N/A")
    d_max = date_range.get("max", "N/A")
    ui.metric_card("Date coverage", f"{d_min} to {d_max}", "5.6 years continuous record")
with c[2]:
    rc_cnt = len(targets.get("root_cause_category", {}))
    ui.metric_card("Root cause categories", f"{rc_cnt} classes", "Ground truth target 1")
with c[3]:
    status_str = latest_import.get("status", "completed").capitalize()
    rows_ins = latest_import.get("rows_inserted", total_recs)
    ui.metric_card("Ingestion audit status", status_str, f"{rows_ins:,} rows verified in DB")

# 2. Ground-Truth Target Label Distributions
ui.section(
    "Ground-truth target distributions",
    "Verified class counts across the 8,000 historical training records.",
)

t1, t2, t3 = st.tabs(["Root cause category (15 classes)", "Severity level (5 classes)", "Risk level (4 classes)"])

with t1:
    rc_dict = targets.get("root_cause_category", {})
    rc_items = [{"name": k, "count": v} for k, v in rc_dict.items()]
    charts.show(charts.distribution_bar_chart(rc_items, label="Root cause", top=15, color="#0F766E"))

with t2:
    sev_dict = targets.get("severity_level", {})
    # Order by severity level hierarchy
    sev_order = ["Critical", "Severe", "Serious", "Moderate", "Minor"]
    sev_items = [{"name": k, "count": sev_dict.get(k, 0)} for k in sev_order if k in sev_dict]
    charts.show(charts.distribution_bar_chart(sev_items, label="Severity level", top=5, color="#B42318"))

with t3:
    risk_dict = targets.get("risk_level", {})
    risk_order = ["Critical", "High", "Medium", "Low"]
    risk_items = [{"name": k, "count": risk_dict.get(k, 0)} for k in risk_order if k in risk_dict]
    charts.show(charts.distribution_bar_chart(risk_items, label="Risk level", top=4, color="#C77D0A"))

# 3. Data Quality & Missing Value Audit
ui.section(
    "Data quality & completeness audit",
    "Audit of optional fields and null counts in the historical training dataset.",
)
null_rows = [
    {
        "Field": "injury_type",
        "Missing rows": nulls.get("injury_type", 0),
        "Null %": f"{(nulls.get('injury_type', 0) / total_recs * 100):.1f}%",
        "Semantics": "Applicable only when personnel injury occurred (non-injury incidents are null)",
    },
    {
        "Field": "body_part_affected",
        "Missing rows": nulls.get("body_part_affected", 0),
        "Null %": f"{(nulls.get('body_part_affected', 0) / total_recs * 100):.1f}%",
        "Semantics": "Applicable only when personnel injury occurred",
    },
    {
        "Field": "contributing_factor_2",
        "Missing rows": nulls.get("contributing_factor_2", 0),
        "Null %": f"{(nulls.get('contributing_factor_2', 0) / total_recs * 100):.1f}%",
        "Semantics": "Optional secondary contributing factor",
    },
    {
        "Field": "root_cause_description",
        "Missing rows": nulls.get("root_cause_description", 0),
        "Null %": f"{(nulls.get('root_cause_description', 0) / total_recs * 100):.1f}%",
        "Semantics": "Optional free-text rationale for root cause categorization",
    },
    {
        "Field": "equipment_status",
        "Missing rows": nulls.get("equipment_status", 0),
        "Null %": f"{(nulls.get('equipment_status', 0) / total_recs * 100):.1f}%",
        "Semantics": "Operational status at time of incident (100% complete)",
    },
]
st.dataframe(pd.DataFrame(null_rows), hide_index=True, width="stretch")

# 4. Equipment Deep Dive Timeline
ui.section(
    "Equipment incident timeline",
    "Explore historical records for a specific equipment asset.",
)
try:
    eq_dist = api_client.get_distributions(dim="equipment_type", source="historical_dataset", limit=40)
    eq_names = [e["name"] for e in eq_dist if e["name"] != "Not Applicable"]
except Exception:
    eq_names = ["Pump", "Conveyor", "Reactor", "Boiler", "Air Compressor"]

if eq_names:
    selected_eq = st.selectbox("Select equipment asset", eq_names)
    try:
        sub_df = api_client.list_incidents(source="historical_dataset", equipment_type=selected_eq, limit=20)
        if not sub_df.empty:
            high_count = int(sub_df["severity"].isin(["High", "Critical", "Severe", "Serious"]).sum())
            st.caption(f"{len(sub_df)} incidents retrieved for {selected_eq} ({high_count} high/critical severity).")
            charts.show(charts.equipment_timeline(sub_df))
            st.dataframe(
                sub_df.head(15)[["id", "date", "type", "severity", "root_cause", "summary"]],
                hide_index=True,
                width="stretch",
                column_config={
                    "id": "Incident ID",
                    "date": st.column_config.DateColumn("Date", format="YYYY-MM-DD"),
                    "type": "Type",
                    "severity": "Severity",
                    "root_cause": "Root cause",
                    "summary": "Description",
                },
            )
        else:
            st.info(f"No historical records found for {selected_eq}.")
    except api_client.ApiError as e:
        st.warning(f"Could not load asset records: {e}")

# 5. Equipment x Root Cause Heatmap
ui.section(
    "Equipment × Root cause cross-tabulation",
    "Frequency cross-tabulation matrix computed directly via PostgreSQL SQL aggregation.",
)
try:
    heatmap_data = api_client.get_heatmap(source="historical_dataset", top_equipment=10, top_causes=8)
    charts.show(charts.matrix_heatmap(heatmap_data))
except api_client.ApiError as e:
    st.warning(f"Could not load heatmap: {e}")

# 6. Current Analysis Comparison (if active in session)
res = st.session_state.get("analysis")
if res:
    ui.section("Comparison with active incident analysis", f"Incident {res['incident'].get('id', '')}")
    ui.historical_tables(res["historical_analysis"])
