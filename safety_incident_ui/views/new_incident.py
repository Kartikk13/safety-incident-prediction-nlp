import streamlit as st

from components import ui
from data.mock_data import EXAMPLE_REPORT  # demo convenience only
from services import api_client
from utils import validation

import hashlib
import streamlit as st

from datetime import datetime


ui.page_header("New incident analysis", "Provide an incident report and the system will extract key details, suggest probable root causes, and compare with history.")
ui.demo_notice()

def _load_example():
    st.session_state["report_text"] = EXAMPLE_REPORT

method = st.radio(
    "How would you like to report the incident?",
    ["Upload / paste report", "Fill short form"],
    horizontal=True,
)

text, err = "", None
analyze_clicked = False
short_form_submitted = False

if method == "Upload / paste report":
    source = st.radio(
        "Report source",
        ["Paste report text", "Upload document"],
        horizontal=True,
    )

    if source == "Paste report text":
        st.session_state.pop("uploaded_filename", None)
        st.session_state.pop("uploaded_file_bytes", None)
        st.session_state.pop("uploaded_mime_type", None)
        st.session_state.pop("uploaded_storage_key", None)
        st.session_state.pop("uploaded_storage_signature", None)

        st.text_area(
            "Incident report",
            key="report_text",
            height=220,
            placeholder=(
                "Describe what happened, when and where, "
                "which equipment was involved, and any contributing circumstances."
            ),
        )
        st.button("Use an example report", on_click=_load_example)
        text = st.session_state.get("report_text", "")

    else:
        exts = ", ".join(validation.SUPPORTED_EXTENSIONS)
        f = st.file_uploader(
            f"Incident report file ({exts}, up to {validation.MAX_FILE_MB} MB)",
            type=[e.lstrip(".") for e in validation.SUPPORTED_EXTENSIONS],
        )
        if f is not None:
            file_bytes = f.getvalue()
            ok, msg, text = validation.validate_upload(f.name, file_bytes)

            if not ok:
                err = msg
                st.session_state.pop("uploaded_filename", None)
                st.session_state.pop("uploaded_file_bytes", None)
                st.session_state.pop("uploaded_mime_type", None)
                st.session_state.pop("uploaded_storage_key", None)
                st.session_state.pop("uploaded_storage_signature", None)
            else:
                st.session_state["uploaded_filename"] = f.name
                st.session_state["uploaded_file_bytes"] = file_bytes
                st.session_state["uploaded_mime_type"] = f.type

                file_signature = hashlib.sha256(
                    f.name.encode("utf-8") + b"\0" + file_bytes
                ).hexdigest()

                if st.session_state.get("uploaded_storage_signature") != file_signature:
                    st.session_state.pop("uploaded_storage_key", None)

                st.session_state["uploaded_storage_signature"] = file_signature

        else:
            st.session_state.pop("uploaded_filename", None)
            st.session_state.pop("uploaded_file_bytes", None)
            st.session_state.pop("uploaded_mime_type", None)
            st.session_state.pop("uploaded_storage_key", None)
            st.session_state.pop("uploaded_storage_signature", None)

else:
    st.caption(
        "Enter the details you know. The NLP stage will extract additional "
        "information from your description; missing details must not be guessed."
    )

    st.session_state.pop("uploaded_filename", None)
    st.session_state.pop("uploaded_file_bytes", None)
    st.session_state.pop("uploaded_mime_type", None)
    st.session_state.pop("uploaded_storage_key", None)
    st.session_state.pop("uploaded_storage_signature", None)

    with st.form("short_incident_form"):
        incident_date = st.date_input("Incident date")
        incident_time = st.time_input("Incident time")
        department = st.text_input("Department (if known)")
        location = st.text_input("Location / plant / unit (if known)")
        shift = st.selectbox(
            "Shift (if known)",
            ["Unknown", "Morning", "Afternoon", "Evening", "Night"],
        )
        description = st.text_area(
            "What happened?",
            height=180,
            placeholder=(
                "Describe the incident, equipment involved, hazard, "
                "what the worker was doing, and any known consequences."
            ),
        )
        short_form_submitted = st.form_submit_button("Analyze incident", type="primary")

    if short_form_submitted:

        context = [
            f"Incident date: {incident_date.isoformat()}",
            f"Incident time: {incident_time.strftime('%H:%M')}",
        ]
        if department.strip():
            context.append(f"Department: {department.strip()}")
        if location.strip():
            context.append(f"Location: {location.strip()}")
        if shift != "Unknown":
            context.append(f"Shift: {shift}")

        text = "\n".join(context) + "\nIncident description: " + description.strip()
        st.session_state["short_form_text"] = text

    text = st.session_state.get("short_form_text", "")

if method == "Upload / paste report":
    analyze_clicked = st.button("Analyze incident", type="primary")

if analyze_clicked or short_form_submitted:
    ok, msg = (False, err) if err else validation.validate_report_text(text)

    if not ok:
        st.error(msg or "Provide a report to analyze.")
    else:
        try:
            with st.status("Extracting incident details...", expanded=True) as status:
                from services.nlp_extractor import extract_incident_features

                context = {}

                if method == "Fill short form":
                    context = {
                        "department": department.strip() or None,
                        "location": location.strip() or None,
                        "shift": None if shift == "Unknown" else shift,
                    }

                
                extracted = extract_incident_features(text, context=context)

                # 4-tier precedence:
                # 1. User corrections (post-extraction review form)
                # 2. Explicit mentions & domain aliases with negation/resolution guards
                # 3. Confident semantic matching
                # 4. Conservative fallback (None)
                try:
                    from hybrid_nlp_prototype import extract_hybrid_features
                    hybrid = extract_hybrid_features(text)
                    for field in (
                        "equipment_type",
                        "equipment_status",
                        "hazard_type",
                        "department",
                    ):
                        hybrid_diag = hybrid.get("diagnostics", {}).get(field, {})
                        # If hybrid identified a negation or resolution guard, enforce None
                        if hybrid_diag.get("method") == "rule_guard":
                            extracted[field] = None
                            extracted["nlp_diagnostics"][field] = hybrid_diag
                        # If hybrid identified an explicit alias or accepted semantic match
                        elif hybrid.get(field) is not None:
                            extracted[field] = hybrid[field]
                            extracted["nlp_diagnostics"][field] = hybrid_diag

                    extracted["nlp_method"] = "hybrid_rules_and_prototype"
                    extracted["nlp_review_required"] = hybrid.get("review_required", True)
                except Exception:
                    # Robust fallback to production rule-based extractor
                    extracted["nlp_method"] = "production_hybrid_rules"
                    extracted["nlp_review_required"] = True

                
                if method == "Fill short form" and short_form_submitted:
                    extracted["incident_year"] = incident_date.year
                    extracted["incident_month"] = incident_date.month
                    extracted["incident_dayofweek"] = incident_date.weekday()
                    extracted["incident_hour"] = incident_time.hour
                    extracted["incident_minute"] = incident_time.minute

                status.update(
                    label="Feature extraction complete",
                    state="complete",
                    expanded=False,
                )

            st.session_state["extracted_features"] = extracted
            st.session_state["source_report_text"] = text
            st.session_state["show_feature_review"] = True
            st.rerun()

        except Exception as e:
            st.error(f"Feature extraction failed: {e}")

elif not text:
    st.info(
        "Paste a report or upload a file, then select Analyze incident. "
        "You do not need to fill in any other fields."
    )

if st.session_state.get("show_feature_review", False):
    st.divider()
    st.subheader("Review extracted incident details")
    st.caption(
        "Check the extracted values and correct them before prediction. "
        "Leave unknown fields blank; do not guess."
    )

    features = st.session_state["extracted_features"]

    with st.container():
        st.markdown("**Incident date and time**")

        col1, col2, col3 = st.columns(3)

        with col1:
            year_known = st.checkbox(
                "Year is known",
                value=features.get("incident_year") is not None,
            )
            year = None
            if year_known:
                year = st.number_input(
                    "Year",
                    min_value=2000,
                    max_value=2100,
                    value=int(features["incident_year"])
                    if features.get("incident_year") is not None
                    else 2026,
                )

        with col2:
            month_known = st.checkbox(
                "Month is known",
                value=features.get("incident_month") is not None,
            )
            month = None
            if month_known:
                month = st.selectbox(
                    "Month",
                    options=list(range(1, 13)),
                    index=(
                        int(features["incident_month"]) - 1
                        if features.get("incident_month") is not None
                        else 0
                    ),
                )

        with col3:
            weekdays = [
                "Monday", "Tuesday", "Wednesday", "Thursday",
                "Friday", "Saturday", "Sunday",
            ]
            day_known = st.checkbox(
                "Day of week is known",
                value=features.get("incident_dayofweek") is not None,
            )
            dayofweek = None
            if day_known:
                dayofweek = st.selectbox(
                    "Day of week",
                    options=weekdays,
                    index=(
                        int(features["incident_dayofweek"])
                        if features.get("incident_dayofweek") is not None
                        else 0
                    ),
                )

        col1, col2 = st.columns(2)

        with col1:
            hour_known = st.checkbox(
                "Hour is known",
                value=features.get("incident_hour") is not None,
            )
            hour = None
            if hour_known:
                hour = st.number_input(
                    "Hour (0-23)",
                    min_value=0,
                    max_value=23,
                    value=int(features["incident_hour"])
                    if features.get("incident_hour") is not None
                    else 0,
                )

        with col2:
            minute_known = st.checkbox(
                "Minute is known",
                value=features.get("incident_minute") is not None,
            )
            minute = None
            if minute_known:
                minute = st.number_input(
                    "Minute",
                    min_value=0,
                    max_value=59,
                    value=int(features["incident_minute"])
                    if features.get("incident_minute") is not None
                    else 0,
                )

        st.markdown("**Workplace and personnel**")

        department = st.selectbox(
            "Department",
            ["Unknown"] + CATEGORIES["department"],
        ) if False else st.text_input(
            "Department", value=features.get("department") or ""
        )

        location = st.text_input(
            "Location", value=features.get("location") or ""
        )
        shift = st.selectbox(
            "Shift",
            ["Unknown", "Morning", "Evening", "Night"],
            index=(
                ["Unknown", "Morning", "Evening", "Night"].index(
                    features["shift"]
                )
                if features.get("shift") in ["Morning", "Evening", "Night"]
                else 0
            ),
        )
        employee_role = st.text_input(
            "Employee role", value=features.get("employee_role") or ""
        )
        experience_level = st.selectbox(
            "Experience level",
            [
                "Unknown", "Less than 1 year", "1-3 years",
                "3-7 years", "More than 7 years",
            ],
            index=(
                [
                    "Unknown", "Less than 1 year", "1-3 years",
                    "3-7 years", "More than 7 years",
                ].index(features["experience_level"])
                if features.get("experience_level") in [
                    "Less than 1 year", "1-3 years",
                    "3-7 years", "More than 7 years",
                ]
                else 0
            ),
        )
        reported_by_role = st.text_input(
            "Reported by role", value=features.get("reported_by_role") or ""
        )

        st.markdown("**Equipment and incident**")

        equipment_type = st.text_input(
            "Equipment type", value=features.get("equipment_type") or ""
        )
        equipment_status = st.text_input(
            "Equipment status", value=features.get("equipment_status") or ""
        )
        incident_type = st.text_input(
            "Incident type", value=features.get("incident_type") or ""
        )
        hazard_type = st.text_input(
            "Hazard type", value=features.get("hazard_type") or ""
        )
        incident_description = st.text_area(
            "Incident description",
            value=features.get("incident_description") or "",
            height=160,
        )

        review_submitted = st.button(
            "Confirm details",
            type="primary",
            key="confirm_reviewed_details",
        )

    if review_submitted:
        reviewed = {
            "incident_year": int(year) if year is not None else None,
            "incident_month": int(month) if month is not None else None,
            "incident_dayofweek": (
                weekdays.index(dayofweek) if dayofweek is not None else None
            ),
            "incident_hour": int(hour) if hour is not None else None,
            "incident_minute": int(minute) if minute is not None else None,
            "department": department.strip() or None,
            "location": location.strip() or None,
            "shift": None if shift == "Unknown" else shift,
            "employee_role": employee_role.strip() or None,
            "experience_level": (
                None if experience_level == "Unknown"
                else experience_level
            ),
            "equipment_type": equipment_type.strip() or None,
            "equipment_status": equipment_status.strip() or None,
            "incident_type": incident_type.strip() or None,
            "hazard_type": hazard_type.strip() or None,
            "incident_description": incident_description.strip(),
            "reported_by_role": reported_by_role.strip() or None,
        }

        
        st.session_state["reviewed_features"] = reviewed

        if not reviewed["incident_description"].strip():
            st.error("Incident description cannot be empty.")
        else:
            try:
                with st.spinner("Generating predictions..."):
                    
                    original_report = st.session_state.get("source_report_text", "")

                    if method == "Fill short form":
                        submission_method = "form"
                    elif st.session_state.get("uploaded_filename"):
                        submission_method = "upload"
                    else:
                        submission_method = "paste"

                    submission_payload = {
                        "reviewed_features": reviewed,
                        "nlp_diagnostics": st.session_state.get("extracted_features", {}).get("nlp_diagnostics", {}),
                        "nlp_method": st.session_state.get("extracted_features", {}).get("nlp_method", "production_hybrid_rules"),
                    }

                    uploaded_file_key = None

                    if submission_method == "form":
                        submission_payload["form_text"] = original_report

                    
                    elif submission_method == "upload":
                        filename = st.session_state["uploaded_filename"]
                        file_bytes = st.session_state.get("uploaded_file_bytes")
                        file_signature = st.session_state.get("uploaded_storage_signature")

                        if not file_bytes or not file_signature:
                            raise api_client.ApiError(
                                "Uploaded file is unavailable. Please upload it again."
                            )

                        uploaded_file_key = st.session_state.get("uploaded_storage_key")

                        if not uploaded_file_key:
                            file_metadata = api_client.upload_incident_file(
                                file_bytes=file_bytes,
                                original_filename=filename,
                                mime_type=st.session_state.get("uploaded_mime_type"),
                            )

                            uploaded_file_key = file_metadata["storage_key"]
                            st.session_state["uploaded_storage_key"] = uploaded_file_key

                        submission_payload["original_filename"] = filename
                        submission_payload["mime_type"] = st.session_state.get(
                            "uploaded_mime_type", "application/octet-stream"
                        )


                    predictions = api_client.predict_incident(
                        reviewed,
                        original_report_text=original_report,
                        submission_method=submission_method,
                        submission_payload=submission_payload,
                        uploaded_file_key=uploaded_file_key,
                    )

                    incident_id = predictions.get("incident_id")
                    if incident_id:
                        st.session_state["current_incident_id"] = incident_id
                        st.query_params["id"] = incident_id
                        try:
                            st.session_state["analysis"] = api_client.get_incident_analysis(incident_id)
                        except Exception:
                            pass

                st.session_state["incident_predictions"] = predictions
                st.session_state["prediction_complete"] = True
                st.success("Predictions generated successfully and saved to PostgreSQL.")
                
                st.subheader("Prediction results")

                col1, col2, col3 = st.columns([1.5, 1.1, 1], gap="medium")

                with col1:
                    st.metric(
                        label="Probable root cause",
                        value=predictions["root_cause_category"],
                        height="content",
                    )

                with col2:
                    st.metric(
                        label="Predicted severity",
                        value=predictions["severity_level"],
                        height="content",
                    )

                with col3:
                    st.metric(
                        label="Predicted risk",
                        value=predictions["risk_level"],
                        height="content",
                    )

                st.caption(
                    "These are model-generated predictions, not confirmed findings. "
                    "Review them before making safety decisions."
                )

                if incident_id:
                    st.divider()
                    b_col, t_col = st.columns([1, 1.5])
                    with b_col:
                        if st.button("Open incident analysis", type="primary", key="btn_open_full_analysis"):
                            st.switch_page("views/analysis.py")
                    with t_col:
                        st.caption(f"Incident saved to database: `{incident_id}`")

            except api_client.ApiError as e:
                st.error(f"Prediction failed: {e}")
