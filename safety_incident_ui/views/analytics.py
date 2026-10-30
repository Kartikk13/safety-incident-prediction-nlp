import pandas as pd
import streamlit as st

from components import charts, ui
from services import api_client

if st.session_state.get("user", {}).get("role") != "analyst":
    st.warning("This page is reserved for safety analysts and administrators."); st.stop()

ui.page_header(
    "Safety analytics",
    "Population breakdowns, multidimensional distributions, and ML model evaluation.",
)
ui.demo_notice()

# 1. Population Filter
pop_choice = st.radio(
    "Analysis population",
    ["All records", "User submissions only", "Historical dataset only"],
    horizontal=True,
    key="analyst_pop_filter",
)
source_param = None
if pop_choice == "User submissions only":
    source_param = "user_submission"
elif pop_choice == "Historical dataset only":
    source_param = "historical_dataset"

try:
    with st.spinner("Fetching safety analytics KPIs..."):
        kpis = api_client.get_kpis(source=source_param)
except api_client.ApiError as e:
    st.error(str(e))
    st.stop()

# 2. Population Metrics
c = st.columns(4)
with c[0]:
    ui.metric_card("Total in population", f"{kpis.get('total_incidents', 0):,}")
with c[1]:
    h_cnt = kpis.get("high_severity_count", 0)
    h_pct = kpis.get("high_severity_pct", 0.0)
    ui.metric_card("High / Critical severity", f"{h_cnt:,}", f"{h_pct:.1f}% of population")
with c[2]:
    gt_cnt = kpis.get("ground_truth_count", 0)
    ui.metric_card("Ground-truth labeled", f"{gt_cnt:,}", "Verified historical records")
with c[3]:
    pr_cnt = kpis.get("predicted_count", 0)
    ui.metric_card("Model predictions", f"{pr_cnt:,}", "User-submitted prediction runs")

# 3. Multidimensional Distributions
ui.section(
    "Organizational & operational distributions",
    f"Breakdowns across departments, locations, shifts, and hazards for {pop_choice.lower()}.",
)

r1_a, r1_b = st.columns(2)
with r1_a:
    try:
        dept_dist = api_client.get_distributions(dim="department", source=source_param, limit=10)
        charts.show(charts.distribution_bar_chart(dept_dist, label="Department", top=10, color="#0F766E"))
    except api_client.ApiError as e:
        st.warning(f"Department distribution unavailable: {e}")

with r1_b:
    try:
        loc_dist = api_client.get_distributions(dim="location", source=source_param, limit=10)
        charts.show(charts.distribution_bar_chart(loc_dist, label="Location / Plant unit", top=10, color="#2F5D8A"))
    except api_client.ApiError as e:
        st.warning(f"Location distribution unavailable: {e}")

r2_a, r2_b = st.columns(2)
with r2_a:
    try:
        shift_dist = api_client.get_distributions(dim="shift", source=source_param, limit=5)
        charts.show(charts.distribution_bar_chart(shift_dist, label="Operating shift", top=5, color="#C77D0A"))
    except api_client.ApiError as e:
        st.warning(f"Shift distribution unavailable: {e}")

with r2_b:
    try:
        hazard_dist = api_client.get_distributions(dim="hazard_type", source=source_param, limit=8)
        charts.show(charts.distribution_bar_chart(hazard_dist, label="Hazard classification", top=8, color="#B42318"))
    except api_client.ApiError as e:
        st.warning(f"Hazard distribution unavailable: {e}")

# 4. Actual vs Predicted Distributions (Explicitly Separated)
ui.section(
    "Ground truth vs Model predictions",
    "Comparison between actual verified labels (historical dataset) and model inferences (user submissions).",
)
st.caption(
    "Note: Actual labels represent ground truth from past investigations. "
    "Predictions represent ML model outputs on user submissions. These are distinct populations."
)

tab_rc, tab_sev, tab_risk = st.tabs(["Root cause category", "Severity classification", "Risk level"])

with tab_rc:
    c_act, c_pred = st.columns(2)
    with c_act:
        st.markdown("**Actual root causes (Historical ground truth)**")
        try:
            rc_act = api_client.get_distributions(dim="root_cause", source="historical_dataset", limit=10)
            charts.show(charts.distribution_bar_chart(rc_act, label="Actual root cause", top=10, color="#0F766E"))
        except Exception as e:
            st.info(f"Actual causes: {e}")
    with c_pred:
        st.markdown("**Predicted root causes (User submission models)**")
        try:
            rc_pred = api_client.get_distributions(dim="predicted_root_cause", limit=10)
            charts.show(charts.distribution_bar_chart(rc_pred, label="Predicted root cause", top=10, color="#2F5D8A"))
        except Exception as e:
            st.info(f"Predicted causes: {e}")

with tab_sev:
    c_act, c_pred = st.columns(2)
    with c_act:
        st.markdown("**Actual severity levels (Historical ground truth)**")
        try:
            sev_act = api_client.get_distributions(dim="severity", source="historical_dataset", limit=5)
            charts.show(charts.distribution_bar_chart(sev_act, label="Actual severity", top=5, color="#B42318"))
        except Exception as e:
            st.info(f"Actual severity: {e}")
    with c_pred:
        st.markdown("**Predicted severity levels (User submission models)**")
        try:
            sev_pred = api_client.get_distributions(dim="predicted_severity", limit=5)
            charts.show(charts.distribution_bar_chart(sev_pred, label="Predicted severity", top=5, color="#C77D0A"))
        except Exception as e:
            st.info(f"Predicted severity: {e}")

with tab_risk:
    c_act, c_pred = st.columns(2)
    with c_act:
        st.markdown("**Actual risk levels (Historical ground truth)**")
        try:
            risk_act = api_client.get_distributions(dim="risk", source="historical_dataset", limit=4)
            charts.show(charts.distribution_bar_chart(risk_act, label="Actual risk", top=4, color="#C77D0A"))
        except Exception as e:
            st.info(f"Actual risk: {e}")
    with c_pred:
        st.markdown("**Predicted risk levels (User submission models)**")
        try:
            risk_pred = api_client.get_distributions(dim="predicted_risk", limit=4)
            charts.show(charts.distribution_bar_chart(risk_pred, label="Predicted risk", top=4, color="#0F766E"))
        except Exception as e:
            st.info(f"Predicted risk: {e}")

# 5. Model Evaluation & Governance Section
ui.section(
    "ML model evaluation & governance",
    "Model performance accounting and registered pipeline artifacts.",
)

try:
    eval_metrics = api_client.get_eval_metrics()
    ev_count = eval_metrics.get("evaluated_records", 0)

    c1, c2, c3 = st.columns(3)
    with c1:
        ui.metric_card("Validation overlap records", f"{ev_count:,}", "Records with both ground-truth and prediction")
    with c2:
        ui.metric_card("Historical training records", f"{eval_metrics.get('total_historical_with_ground_truth', 8000):,}", "Ground truth baseline")
    with c3:
        ui.metric_card("User submission predictions", f"{eval_metrics.get('total_prediction_runs', 0):,}", "Live inference executions")

    st.info(
        eval_metrics.get(
            "methodology_note",
            "Model accuracy requires independent validation incidents with verified ground-truth labels and model predictions. "
            "Training dataset records are excluded from evaluation metrics to prevent optimistic overfitting bias.",
        )
    )

    if ev_count > 0:
        st.subheader("Model validation accuracy")
        acc = eval_metrics.get("accuracy", {})
        acc_cols = st.columns(3)
        with acc_cols[0]:
            ui.metric_card("Root cause accuracy", f"{acc.get('root_cause_category', 0.0):.1%}")
        with acc_cols[1]:
            ui.metric_card("Severity level accuracy", f"{acc.get('severity_level', 0.0):.1%}")
        with acc_cols[2]:
            ui.metric_card("Risk level accuracy", f"{acc.get('risk_level', 0.0):.1%}")
    else:
        st.caption("No overlapping evaluation incidents currently recorded. Accuracy calculation will activate when ground-truth labels are confirmed on user submissions.")

except api_client.ApiError as e:
    st.warning(f"Model evaluation service unavailable: {e}")
