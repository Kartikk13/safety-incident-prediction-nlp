import streamlit as st

from components import ui

st.set_page_config(page_title="Safety Incident Intelligence", page_icon=":material/shield:", layout="wide")
ui.inject_css()

user = st.session_state.get("user")
if not user:
    ui.login_view()
    st.stop()

nav = {"": [
    st.Page("views/dashboard.py", title="Dashboard", icon=":material/dashboard:", default=True),
    st.Page("views/new_incident.py", title="New incident", icon=":material/note_add:"),
    st.Page("views/analysis.py", title="Incident analysis", icon=":material/fact_check:"),
    st.Page("views/historical.py", title="Historical analysis", icon=":material/history:"),
    st.Page("views/history.py", title="Incident history", icon=":material/table_rows:"),
]}
if user["role"] == "analyst":
    nav["Analyst"] = [st.Page("views/analytics.py", title="Safety analytics", icon=":material/monitoring:")]

with st.sidebar:
    st.markdown("**Safety Incident Intelligence**")
    st.caption(f"{user['name']} ({'Analyst' if user['role'] == 'analyst' else 'User'})")
    if st.button("Sign out"):
        st.session_state.clear()
        st.rerun()

st.navigation(nav).run()
