"""
utils/progress_manager.py

Enterprise Progress Manager
"""

from __future__ import annotations

import streamlit as st


# ==========================================================
# Initialize
# ==========================================================

def initialize():

    defaults = {

        "progress": 0,

        "progress_text": "Waiting...",

        "processing": False,

        "rag_ready": False,

    }

    for key, value in defaults.items():

        if key not in st.session_state:

            st.session_state[key] = value


# ==========================================================
# Update Progress
# ==========================================================

def update_progress(
    value: int,
    message: str,
):

    initialize()

    st.session_state.progress = value

    st.session_state.progress_text = message

    st.session_state.processing = True


# ==========================================================
# Complete
# ==========================================================

def complete():

    initialize()

    st.session_state.progress = 100

    st.session_state.progress_text = "Knowledge Base Ready"

    st.session_state.processing = False

    st.session_state.rag_ready = True


# ==========================================================
# Reset
# ==========================================================

def reset():

    initialize()

    st.session_state.progress = 0

    st.session_state.progress_text = "Waiting..."

    st.session_state.processing = False

    st.session_state.rag_ready = False


# ==========================================================
# Get Status
# ==========================================================

def get_progress():

    initialize()

    return {

        "progress": st.session_state.progress,

        "message": st.session_state.progress_text,

        "processing": st.session_state.processing,

        "ready": st.session_state.rag_ready,

    }