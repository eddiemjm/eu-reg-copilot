"""Streamlit UI.  Run:  streamlit run app.py

The pages live in ui/views/; this file sets up the theme, the shared sidebar and the navigation.
"""
from __future__ import annotations

import streamlit as st

from ui import components, settings, theme

st.set_page_config(page_title="EU AI Act & DORA Copilot", page_icon=theme.favicon(), layout="wide")
theme.inject()

s = settings.sidebar()
st.session_state["settings"] = s
kpis, kpi_source = components.header_kpis(s)
theme.header(s.mode_label, s.local, kpis, kpi_source)

st.navigation([
    st.Page("ui/views/ask.py", title="Ask", icon=":material/forum:", default=True),
    st.Page("ui/views/compare.py", title="Compare models", icon=":material/compare_arrows:"),
    st.Page("ui/views/evaluation.py", title="Evaluation", icon=":material/monitoring:"),
    st.Page("ui/views/how.py", title="How it works", icon=":material/account_tree:"),
], position="top").run()
