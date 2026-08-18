"""
streamlit_app.py

ONGC Enterprise AI - Chat Interface

Run with:
    streamlit run streamlit_app.py
"""

from __future__ import annotations

import base64
import json
import uuid
from pathlib import Path

import streamlit as st
import streamlit.components.v1 as components

from config import APP_TITLE, DATA_DIR, PROJECT_ROOT
from backend.utils.logger import get_logger
from core.rag_engine import build_rag, ask_question

logger = get_logger(__name__)

UPLOAD_DIR = Path(DATA_DIR)
UPLOAD_DIR.mkdir(parents=True, exist_ok=True)

HISTORY_FILE = Path(PROJECT_ROOT) / "chat_history.json"

LOGO_PATH = Path(PROJECT_ROOT) / "assets" / "ongc_logo.png"


def load_conversations() -> dict:
    if HISTORY_FILE.exists():
        try:
            with open(HISTORY_FILE, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception:
            logger.exception("Failed to load conversation history.")
    return {}


def save_conversations():
    try:
        data = {
            cid: c
            for cid, c in st.session_state.conversations.items()
            if c["messages"]
        }
        with open(HISTORY_FILE, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
    except Exception:
        logger.exception("Failed to save conversation history.")


st.set_page_config(
    page_title=APP_TITLE,
    page_icon=str(LOGO_PATH) if LOGO_PATH.exists() else "🛢️",
    layout="wide",
    initial_sidebar_state="expanded",
)


@st.cache_data(show_spinner=False)
def _logo_base64() -> str:
    with open(LOGO_PATH, "rb") as f:
        return base64.b64encode(f.read()).decode("utf-8")


def ongc_logo_img(size: int = 28) -> str:
    return (
        f'<img src="data:image/png;base64,{_logo_base64()}" '
        f'width="{size}" height="{size}" '
        f'style="border-radius:50%;object-fit:cover;display:block;" />'
    )


def init_state():
    if "conversations" not in st.session_state:
        st.session_state.conversations = load_conversations()

    defaults = {
        "active_conv_id": None,
        "uploaded_names": [],
        "kb_ready": False,
        "rag": None,
        "processing": False,
        "process_status": None,
        "process_pct": 0,
        "pending_question": None,
        "uploader_version": 0,
    }
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value

    if st.session_state.active_conv_id is None:
        new_chat()


def new_chat():
    conv_id = str(uuid.uuid4())
    st.session_state.conversations[conv_id] = {
        "title": "New Chat",
        "messages": [],
    }
    st.session_state.active_conv_id = conv_id


def active_conv():
    return st.session_state.conversations[st.session_state.active_conv_id]


init_state()


st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');

    *, *::before, *::after {
        box-sizing: border-box;
        font-family: 'Inter', sans-serif;
    }

    #MainMenu, footer { visibility: hidden; }
    .stAppDeployButton { display: none !important; }

    header[data-testid="stHeader"] {
        background-color: #0b0b0d;
        height: 0 !important;
        min-height: 0 !important;
    }

    .stApp {
        background-color: #0b0b0d;
        color: #e7e7ea;
    }

    section[data-testid="stSidebar"] {
        background-color: #111113 !important;
        border-right: 1px solid #1f1f23 !important;
        width: 260px !important;
        min-width: 260px !important;
        max-width: 260px !important;
    }

    div[data-testid="stSidebarResizeHandle"] { display: none !important; }

    section[data-testid="stSidebar"] > div:first-child {
        height: 100vh !important;
        display: flex !important;
        flex-direction: column !important;
        overflow: hidden !important;
        padding: 0 !important;
    }

    section[data-testid="stSidebar"] .block-container {
        flex: 1 1 0 !important;
        min-height: 0 !important;
        overflow-y: auto !important;
        overflow-x: hidden !important;
        display: flex !important;
        flex-direction: column !important;
        padding: 14px 12px 0 12px !important;
        gap: 0 !important;
        overscroll-behavior: contain !important;
    }

    section[data-testid="stSidebar"] .block-container > div,
    section[data-testid="stSidebar"] .block-container > div > div {
        flex-shrink: 0 !important;
    }

    section[data-testid="stSidebar"] .block-container .stVerticalBlock {
        gap: 0 !important;
    }

    .st-key-doc_list_scroll,
    .st-key-doc_list_scroll > div {
        flex-shrink: 0 !important;
    }
    .st-key-doc_list_scroll [data-testid="stVerticalBlockBorderWrapper"] {
        height: 112px !important;
        overflow-y: auto !important;
    }

    .st-key-process_status_zone [data-testid="stVerticalBlockBorderWrapper"] {
        height: 50px !important;
        overflow: hidden !important;
    }

    .st-key-right_panel {
        position: fixed !important;
        top: 0 !important;
        right: 0 !important;
        width: 260px !important;
        min-width: 260px !important;
        max-width: 260px !important;
        height: 100vh !important;
        background-color: #111113 !important;
        border-left: 1px solid #1f1f23 !important;
        z-index: 999 !important;
        display: flex !important;
        flex-direction: column !important;
        overflow: hidden !important;
        padding: 0 !important;
        box-sizing: border-box !important;
        overscroll-behavior: contain !important;
    }
    .st-key-right_panel > div {
        display: flex !important;
        flex-direction: column !important;
        flex: 1 1 auto !important;
        min-height: 0 !important;
        padding: 14px 12px 12px 12px !important;
        box-sizing: border-box !important;
    }
    .rp-label {
        font-size: 0.95rem;
        font-weight: 700;
        letter-spacing: -0.01em;
        text-transform: none;
        color: #f0f0f2;
        margin: 0 0 12px 1px;
        flex-shrink: 0;
    }
    .st-key-right_conv_scroll {
        flex: 1 1 0 !important;
        min-height: 0 !important;
        overflow: hidden !important;
    }
    .st-key-right_conv_scroll[data-testid="stVerticalBlockBorderWrapper"] {
        overflow-y: auto !important;
        background: #18181b !important;
        border: 1px solid #28282d !important;
        border-radius: 10px !important;
        padding: 10px !important;
        box-sizing: border-box !important;
        overscroll-behavior: contain !important;
    }
    .st-key-right_conv_scroll [data-testid="stVerticalBlockBorderWrapper"] {
        height: 100% !important;
        max-height: 100% !important;
        overflow-y: auto !important;
        background: #18181b !important;
        border: 1px solid #28282d !important;
        border-radius: 10px !important;
        padding: 10px !important;
        box-sizing: border-box !important;
        overscroll-behavior: contain !important;
    }
    .rp-footer {
        border-top: 1px solid #1f1f22;
        padding-top: 10px;
        margin-top: 10px;
        flex-shrink: 0 !important;
    }
    .rp-footer div[data-testid="stButton"] button {
        width: 100% !important;
        background: transparent !important;
        border: 1px solid #2a2a2e !important;
        color: #9a9aa0 !important;
        border-radius: 9px !important;
        font-size: 0.83rem !important;
        font-weight: 500 !important;
        padding: 7px 10px !important;
        text-align: left !important;
        justify-content: flex-start !important;
        transition: background .2s, color .2s, border-color .2s !important;
    }
    .rp-footer div[data-testid="stButton"] button:hover {
        background: rgba(226,85,74,0.08) !important;
        border-color: #e2554a !important;
        color: #e2554a !important;
    }

    html, body, .stApp {
        overflow-x: hidden !important;
    }
    div[data-testid="stMain"] {
        overflow-x: hidden !important;
        height: 100vh !important;
        overflow-y: auto !important;
        overscroll-behavior: contain !important;
    }
    div[data-testid="stMain"] .block-container {
        max-width: 100% !important;
        padding: 0 !important;
        box-sizing: border-box !important;
        overflow: visible !important;
    }
    .st-key-main_content_col {
        position: fixed !important;
        top: 0 !important;
        left: 260px !important;
        right: 260px !important;
        bottom: 100px !important;
        width: auto !important;
        max-width: none !important;
        margin: 0 !important;
        overflow-y: auto !important;
        overflow-x: hidden !important;
        box-sizing: border-box !important;
        padding: 24px 0 0 0 !important;
    }

    div[data-testid="stBottom"] {
        right: 260px !important;
        left: 260px !important;
        width: auto !important;
    }
    div[data-testid="stBottom"] > div {
        max-width: 850px !important;
        width: 100% !important;
        margin: 0 auto !important;
        padding-left: 16px !important;
        padding-right: 16px !important;
        box-sizing: border-box !important;
    }

    div[data-testid="stChatMessage"] {
        max-width: 850px !important;
        width: 100% !important;
        margin-left: auto !important;
        margin-right: auto !important;
        box-sizing: border-box !important;
    }
    div[data-testid="stChatMessage"] .stMarkdown {
        overflow-wrap: anywhere !important;
        word-break: break-word !important;
    }
    .st-key-main_content_col pre,
    .st-key-main_content_col code {
        white-space: pre-wrap !important;
        word-break: break-word !important;
        overflow-wrap: anywhere !important;
    }
    .st-key-main_content_col table {
        display: block !important;
        max-width: 100% !important;
        overflow-x: auto !important;
    }
    .st-key-main_content_col img {
        max-width: 100% !important;
        height: auto !important;
    }

    .sb-brand {
        display: flex;
        align-items: center;
        gap: 9px;
        padding: 2px 0 14px 0;
        border-bottom: 1px solid #1f1f23;
        margin-bottom: 4px;
    }
    .sb-brand-badge {
        width: 28px; height: 28px;
        border-radius: 50%;
        display: flex; align-items: center; justify-content: center;
        flex-shrink: 0;
        box-shadow: 0 0 0 2px #1a1a1d;
        overflow: hidden;
    }
    .sb-brand-badge svg { display: block; }
    .sb-brand-title {
        flex: 1;
        font-size: 0.87rem;
        font-weight: 600;
        color: #f0f0f2;
        white-space: nowrap;
        overflow: hidden;
        text-overflow: ellipsis;
    }
    .sb-brand-edit {
        color: #44444a;
        font-size: 0.9rem;
        cursor: pointer;
        transition: color .2s;
    }
    .sb-brand-edit:hover { color: #aaaaaf; }

    .sb-label {
        font-size: 0.60rem;
        font-weight: 700;
        letter-spacing: 0.10em;
        text-transform: uppercase;
        color: #44444a;
        margin: 14px 0 6px 1px;
    }

    .sb-new-chat div[data-testid="stButton"] button {
        width: 100% !important;
        background: #1c1c20 !important;
        border: 1px solid #2a2a2f !important;
        color: #e0e0e4 !important;
        border-radius: 9px !important;
        font-size: 0.83rem !important;
        font-weight: 500 !important;
        padding: 8px 14px !important;
        margin-top: 8px !important;
        transition: background .2s, border-color .2s !important;
    }
    .sb-new-chat div[data-testid="stButton"] button:hover {
        background: #232327 !important;
        border-color: #3a3a40 !important;
        color: #fff !important;
    }

    section[data-testid="stSidebar"] [data-testid="stFileUploaderDropzone"] {
        background: #18181b !important;
        border: 1px dashed #2e2e34 !important;
        border-radius: 9px !important;
        padding: 2px 6px !important;
        min-height: 0 !important;
    }
    section[data-testid="stSidebar"] [data-testid="stFileUploaderDropzoneInstructions"] {
        display: none !important;
    }
    section[data-testid="stSidebar"] [data-testid="stFileUploaderDropzone"] button {
        width: 100% !important;
        background: transparent !important;
        border: none !important;
        color: #c8c8cc !important;
        font-size: 0.82rem !important;
    }
    section[data-testid="stSidebar"] [data-testid="stFileUploader"] section { padding: 0 !important; }
    section[data-testid="stSidebar"] div[data-testid*="FileUploaderFile"] { display: none !important; }

    .doc-pill {
        display: flex;
        align-items: center;
        gap: 6px;
        background: #18181b;
        border: 1px solid #28282d;
        border-radius: 8px;
        padding: 5px 4px 5px 9px;
        margin-bottom: 5px;
        font-size: 0.78rem;
        color: #b0b0b6;
    }
    .doc-pill-icon { font-size: 0.78rem; flex-shrink: 0; }
    .doc-pill-name {
        flex: 1;
        white-space: nowrap;
        overflow: hidden;
        text-overflow: ellipsis;
        min-width: 0;
    }

    .doc-rm-btn div[data-testid="stButton"] button {
        background: none !important;
        border: none !important;
        color: #44444a !important;
        font-size: 0.75rem !important;
        padding: 0 3px !important;
        min-height: 0 !important;
        height: 20px !important;
        border-radius: 3px !important;
        transition: color .2s !important;
    }
    .doc-rm-btn div[data-testid="stButton"] button:hover {
        color: #e2554a !important;
        background: rgba(226,85,74,.10) !important;
    }

    .sb-process div[data-testid="stButton"] button {
        width: 100% !important;
        background: #1c1c20 !important;
        border: 1px solid #2a2a2f !important;
        color: #e0e0e4 !important;
        border-radius: 9px !important;
        font-size: 0.83rem !important;
        font-weight: 500 !important;
        padding: 8px 14px !important;
        transition: border-color .2s, background .2s !important;
    }
    .sb-process div[data-testid="stButton"] button:hover:not(:disabled) {
        border-color: #c0202c !important;
        color: #fff !important;
    }
    .sb-process div[data-testid="stButton"] button:disabled {
        opacity: 0.35 !important;
    }

    .stProgress > div > div { background-color: #2a2a2e; border-radius: 4px; }
    .stProgress > div > div > div > div {
        background: linear-gradient(90deg, #c0202c, #e8404f);
        border-radius: 4px;
    }

    /* Scrollbar for the Recent Conversations box. The JS layout
       enforcer below finds whichever DOM node is actually the
       scrolling element and tags it directly with .rp-conv-scrollbox,
       so this styling always lands on the right node regardless of
       how Streamlit happens to nest the border-wrapper here (same
       Case A/B uncertainty called out elsewhere in this file). */
    .rp-conv-scrollbox {
        scrollbar-width: thin !important;
        scrollbar-color: #333338 transparent !important;
    }
    .rp-conv-scrollbox::-webkit-scrollbar {
        width: 6px;
    }
    .rp-conv-scrollbox::-webkit-scrollbar-track {
        background: transparent;
    }
    .rp-conv-scrollbox::-webkit-scrollbar-thumb {
        background: #333338;
        border-radius: 6px;
    }
    .rp-conv-scrollbox::-webkit-scrollbar-thumb:hover {
        background: #46464d;
    }

    /* ============================================================
       Recent Conversations — pill-style bordered buttons (matches
       reference mock: each item is its own rounded, bordered box
       with clear spacing between items, not just a flat list with
       a left accent bar).
    ============================================================ */
    div[class*="st-key-rconv_"] { margin-bottom: 8px; }

    div[class*="st-key-rconv_"] div[data-testid="stButton"] button {
        width: 100% !important;
        background: #1c1c20 !important;
        border: 1px solid #2c2c31 !important;
        color: #c7c7cc !important;
        text-align: left !important;
        font-size: 0.83rem !important;
        font-weight: 500 !important;
        padding: 10px 14px !important;
        border-radius: 10px !important;
        white-space: nowrap !important;
        overflow: hidden !important;
        text-overflow: ellipsis !important;
        display: block !important;
        transition: background .15s, color .15s, border-color .15s !important;
    }
    div[class*="st-key-rconv_"] div[data-testid="stButton"] button:hover {
        background: #232327 !important;
        border-color: #3a3a40 !important;
        color: #f0f0f2 !important;
    }
    div[class*="st-key-rconv_active_"] div[data-testid="stButton"] button {
        background: rgba(226,85,74,0.16) !important;
        border: 1px solid #e2554a !important;
        color: #f5f5f6 !important;
        font-weight: 600 !important;
    }

    section[data-testid="stSidebar"] [data-testid="stVerticalBlockBorderWrapper"] {
        border: none !important;
        box-shadow: none !important;
        background: transparent !important;
    }

    .doc-empty-hint {
        font-size: 0.76rem;
        color: #45454a;
        padding: 8px 4px;
    }

    .main-wrap {
        max-width: 850px;
        margin: 0 auto;
        width: 100%;
        padding: 0 16px;
    }

    .landing-wrap {
        text-align: center;
        padding-top: 7vh;
        padding-bottom: 20px;
    }
    .landing-logo {
        width: 80px;
        height: 80px;
        margin: 0 auto 20px auto;
        border-radius: 50%;
        box-shadow: 0 0 0 5px #1a1a1d, 0 8px 32px rgba(192,32,44,0.3);
    }
    .landing-logo svg { display: block; }
    .landing-title {
        font-size: 1.75rem;
        font-weight: 700;
        color: #f4f4f5;
        margin-bottom: 8px;
        letter-spacing: -0.02em;
    }
    .landing-sub {
        color: #7a7a80;
        font-size: 0.93rem;
        margin-bottom: 32px;
    }

    div[data-testid="stChatMessage"] { background: transparent; padding: 4px 0; }

    div[data-testid="stChatMessage"]:has([data-testid="stChatMessageAvatarUser"]) {
        flex-direction: row-reverse;
        justify-content: flex-end;
    }
    div[data-testid="stChatMessage"]:has([data-testid="stChatMessageAvatarUser"]) [data-testid="stChatMessageAvatarUser"] {
        display: none !important;
    }
    div[data-testid="stChatMessage"]:has([data-testid="stChatMessageAvatarUser"]) .stMarkdown {
        background: transparent;
        color: #f4f4f5;
        font-weight: 500;
        text-align: right;
        padding: 4px 2px;
        display: inline-block;
        max-width: 78%;
    }

    div[data-testid="stChatMessage"]:has([data-testid="stChatMessageAvatarAssistant"]) .stMarkdown {
        background: #17171a;
        border: 1px solid #232326;
        border-radius: 18px 18px 18px 4px;
        padding: 10px 16px;
        display: inline-block;
        max-width: 84%;
    }
    div[data-testid="stChatMessage"]:has([data-testid="stChatMessageAvatarAssistant"]) .stMarkdown::before {
        content: "AI";
        display: block;
        font-size: 0.62rem;
        font-weight: 700;
        letter-spacing: 0.1em;
        color: #e2554a;
        margin-bottom: 5px;
    }
    div[data-testid="stChatMessage"] [data-testid="stChatMessageAvatarAssistant"] {
        background-color: #1a1a1d;
        border: 1px solid #2a2a2e;
    }

    div[data-testid="stBottom"] {
        background-color: transparent !important;
        padding-bottom: 8px;
    }
    div[data-testid="stBottom"] > div {
        background-color: transparent !important;
    }
    div[data-testid="stChatInput"] {
        background-color: #15151a;
        border: 1px solid #2a2a2e;
        border-radius: 18px;
        box-shadow: 0 6px 24px rgba(0,0,0,0.4);
        position: relative;
        max-width: 850px !important;
        width: calc(100% - 32px) !important;
        margin: 0 auto !important;
    }
    div[data-testid="stChatInput"]:focus-within {
        border-color: #c0202c;
        box-shadow: 0 0 0 3px rgba(192,32,44,0.18);
    }
    div[data-testid="stChatInput"] textarea {
        padding-left: 3rem !important;
        color: #e7e7ea !important;
        background: transparent !important;
    }
    div[data-testid="stChatInput"] textarea::placeholder {
        color: #55555a !important;
    }
    div[data-testid="stChatInput"]::before {
        content: "📎";
        position: absolute;
        left: 16px;
        top: 50%;
        transform: translateY(-50%) rotate(-20deg);
        opacity: 0.45;
        font-size: 1rem;
        pointer-events: none;
        z-index: 2;
    }
    div[data-testid="stChatInput"] button {
        background-color: #c0202c !important;
        border-radius: 12px !important;
        border: none !important;
        transition: background 0.2s !important;
    }
    div[data-testid="stChatInput"] button:hover {
        background-color: #a81a25 !important;
    }
    div[data-testid="stChatInput"] button svg {
        fill: #ffffff !important;
        color: #ffffff !important;
    }

    .disclaimer {
        text-align: center;
        color: #45454a;
        font-size: 0.72rem;
        margin-top: 4px;
        padding-bottom: 4px;
    }

    ::-webkit-scrollbar { width: 4px; height: 4px; }
    ::-webkit-scrollbar-track { background: transparent; }
    ::-webkit-scrollbar-thumb { background: #2e2e33; border-radius: 4px; }
    ::-webkit-scrollbar-thumb:hover { background: #3e3e44; }
    </style>
    """,
    unsafe_allow_html=True,
)


components.html(
    """
    <script>
    function fixLayout() {
        try {
            const doc = window.parent.document;

            const bottom = doc.querySelector('div[data-testid="stBottom"]');
            if (bottom) {
                bottom.style.setProperty('position', 'fixed', 'important');
                bottom.style.setProperty('left', '260px', 'important');
                bottom.style.setProperty('right', '260px', 'important');
                bottom.style.setProperty('width', 'auto', 'important');
                bottom.style.setProperty('z-index', '1000', 'important');

                const inner = bottom.querySelector(':scope > div');
                if (inner) {
                    inner.style.setProperty('max-width', '850px', 'important');
                    inner.style.setProperty('width', '100%', 'important');
                    inner.style.setProperty('margin', '0 auto', 'important');
                    inner.style.setProperty('box-sizing', 'border-box', 'important');
                }
            }

            const rightPanel = doc.querySelector('.st-key-right_panel');
            if (rightPanel) {
                rightPanel.style.setProperty('position', 'fixed', 'important');
                rightPanel.style.setProperty('top', '0', 'important');
                rightPanel.style.setProperty('right', '0', 'important');
                rightPanel.style.setProperty('width', '260px', 'important');
                rightPanel.style.setProperty('height', '100vh', 'important');
                rightPanel.style.setProperty('z-index', '999', 'important');
            }

            const mainCol = doc.querySelector('.st-key-main_content_col');
            if (mainCol) {
                mainCol.style.setProperty('position', 'fixed', 'important');
                mainCol.style.setProperty('top', '0', 'important');
                mainCol.style.setProperty('left', '260px', 'important');
                mainCol.style.setProperty('right', '260px', 'important');
                mainCol.style.setProperty('bottom', '100px', 'important');
                mainCol.style.setProperty('width', 'auto', 'important');
                mainCol.style.setProperty('max-width', 'none', 'important');
                mainCol.style.setProperty('margin', '0', 'important');
                mainCol.style.setProperty('overflow-y', 'auto', 'important');
                mainCol.style.setProperty('overflow-x', 'hidden', 'important');
                mainCol.style.setProperty('box-sizing', 'border-box', 'important');
            }

            const convKeyEl = doc.querySelector('.st-key-right_conv_scroll');
            const clearBtn = doc.querySelector('.st-key-clear_conv_btn');

            if (convKeyEl && clearBtn) {

                let ancestor = convKeyEl;
                const panelInner = doc.querySelector('.st-key-right_panel > div');
                while (ancestor && panelInner && ancestor !== panelInner) {
                    ancestor.style.setProperty('flex', '1 1 auto', 'important');
                    ancestor.style.setProperty('min-height', '0', 'important');
                    ancestor = ancestor.parentElement;
                    if (!ancestor || !panelInner.contains(ancestor)) break;
                }

                const isBoxItself = convKeyEl.getAttribute('data-testid') === 'stVerticalBlockBorderWrapper';
                const box = isBoxItself
                    ? convKeyEl
                    : (convKeyEl.querySelector('[data-testid="stVerticalBlockBorderWrapper"]') || convKeyEl);

                const boxTop = box.getBoundingClientRect().top;
                const btnTop = clearBtn.getBoundingClientRect().top;
                const available = Math.round(btnTop - boxTop - 20);

                if (available > 40 && available < window.innerHeight) {
                    box.style.setProperty('height', available + 'px', 'important');
                    box.style.setProperty('max-height', available + 'px', 'important');
                    box.style.setProperty('flex', 'none', 'important');
                    if (!isBoxItself) {
                        convKeyEl.style.setProperty('height', available + 'px', 'important');
                        convKeyEl.style.setProperty('flex', 'none', 'important');
                    }
                }

                box.style.setProperty('overflow-y', 'auto', 'important');
                box.style.setProperty('background', '#18181b', 'important');
                box.style.setProperty('border', '1px solid #28282d', 'important');
                box.style.setProperty('border-radius', '10px', 'important');
                box.style.setProperty('box-sizing', 'border-box', 'important');
                box.style.setProperty('padding', '10px', 'important');
                // Tag the real scrolling node directly so the themed
                // scrollbar CSS (.rp-conv-scrollbox) always lands on
                // it, whichever element it turns out to be.
                box.classList.add('rp-conv-scrollbox');

            }
        } catch (e) {
        }
    }

    fixLayout();
    try {
        const observer = new MutationObserver(fixLayout);
        observer.observe(window.parent.document.body, { childList: true, subtree: true });
    } catch (e) {}
    setInterval(fixLayout, 400);
    </script>
    """,
    height=0,
)


with st.sidebar:

    st.markdown(
        f"""
        <div class="sb-brand">
            <div class="sb-brand-badge">{ongc_logo_img(28)}</div>
            <span class="sb-brand-title">ONGC Enterprise AI</span>
            <span class="sb-brand-edit">✎</span>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.markdown('<div class="sb-new-chat">', unsafe_allow_html=True)
    if st.button("＋  New Chat", use_container_width=True, key="new_chat_btn"):
        new_chat()
        st.rerun()
    st.markdown("</div>", unsafe_allow_html=True)

    st.markdown('<div class="sb-label">Documents</div>', unsafe_allow_html=True)

    uploaded = st.file_uploader(
        "Upload PDF",
        type=["pdf"],
        accept_multiple_files=True,
        label_visibility="collapsed",
        key=f"pdf_uploader_{st.session_state.uploader_version}",
    )
    if uploaded:
        changed = False
        for f in uploaded:
            dest = UPLOAD_DIR / f.name
            if f.name not in st.session_state.uploaded_names:
                with open(dest, "wb") as out:
                    out.write(f.getbuffer())
                st.session_state.uploaded_names.append(f.name)
                changed = True
        if changed:
            st.session_state.uploader_version += 1
            st.rerun()

    with st.container(height=112, border=True, key="doc_list_scroll"):
        if not st.session_state.uploaded_names:
            st.markdown(
                '<div class="doc-empty-hint">No documents uploaded yet</div>',
                unsafe_allow_html=True,
            )
        for name in list(st.session_state.uploaded_names):
            col_name, col_rm = st.columns([0.83, 0.17])
            with col_name:
                st.markdown(
                    f'<div class="doc-pill">'
                    f'<span class="doc-pill-icon">📄</span>'
                    f'<span class="doc-pill-name" title="{name}">{name}</span>'
                    f'</div>',
                    unsafe_allow_html=True,
                )
            with col_rm:
                st.markdown('<div class="doc-rm-btn">', unsafe_allow_html=True)
                if st.button("✕", key=f"rm_{name}"):
                    st.session_state.uploaded_names.remove(name)
                    p = UPLOAD_DIR / name
                    if p.exists():
                        p.unlink()
                    st.rerun()
                st.markdown("</div>", unsafe_allow_html=True)

    st.markdown('<div class="sb-label">Process Documents</div>', unsafe_allow_html=True)

    st.markdown('<div class="sb-process">', unsafe_allow_html=True)
    process_clicked = st.button(
        "▷  Process Documents",
        use_container_width=True,
        disabled=(len(st.session_state.uploaded_names) == 0) or st.session_state.processing,
        key="process_docs_btn",
    )
    st.markdown("</div>", unsafe_allow_html=True)

    with st.container(height=50, border=False, key="process_status_zone"):
        progress_bar_slot = st.empty()
        status_slot = st.empty()

        if process_clicked and st.session_state.uploaded_names:
            st.session_state.processing = True
            st.session_state.kb_ready = False
            bar = progress_bar_slot.progress(0)

            def on_progress(pct: int, message: str):
                bar.progress(min(max(pct, 0), 100))
                status_slot.markdown(
                    f"<div style='font-size:0.75rem;color:#7a7a80;margin-top:4px;'>🔵 {message}</div>",
                    unsafe_allow_html=True,
                )

            try:
                pdf_paths = [str(UPLOAD_DIR / n) for n in st.session_state.uploaded_names]
                rag = build_rag(pdf_paths=pdf_paths, progress_callback=on_progress)
                st.session_state.rag = rag
                st.session_state.kb_ready = True
                status_slot.markdown(
                    "<div style='font-size:0.75rem;color:#4caf6b;margin-top:4px;'>✓ Knowledge Base Ready</div>",
                    unsafe_allow_html=True,
                )
            except Exception:
                logger.exception("Failed to build RAG pipeline.")
                status_slot.markdown(
                    "<div style='font-size:0.75rem;color:#e2554a;margin-top:4px;'>✕ Failed to process documents.</div>",
                    unsafe_allow_html=True,
                )
            finally:
                st.session_state.processing = False

        elif st.session_state.kb_ready:
            status_slot.markdown(
                "<div style='font-size:0.75rem;color:#4caf6b;margin-top:4px;'>✓ Knowledge Base Ready</div>",
                unsafe_allow_html=True,
            )


with st.container(key="right_panel"):

    st.markdown('<div class="rp-label">Recent Conversations</div>', unsafe_allow_html=True)

    with st.container(border=True, key="right_conv_scroll"):
        conv_ids = [
            cid for cid, c in st.session_state.conversations.items() if c["messages"]
        ][::-1]

        active_id = st.session_state.active_conv_id
        if active_id in st.session_state.conversations and active_id not in conv_ids:
            conv_ids.insert(0, active_id)

        if not conv_ids:
            st.markdown(
                "<div style='font-size:0.78rem;color:#45454a;padding:6px 4px;'>No conversations yet</div>",
                unsafe_allow_html=True,
            )

        for cid in conv_ids:
            c_item = st.session_state.conversations[cid]
            label = c_item["title"] if c_item["messages"] else "New Chat"
            is_active = cid == active_id
            row_key = f"rconv_active_{cid}" if is_active else f"rconv_{cid}"
            with st.container(key=row_key):
                if st.button(label, key=f"rconv_btn_{cid}", use_container_width=True):
                    st.session_state.active_conv_id = cid
                    st.rerun()

    st.markdown('<div class="rp-footer">', unsafe_allow_html=True)
    if st.button("🗑  Clear Conversations", use_container_width=True, key="clear_conv_btn"):
        st.session_state.conversations = {}
        save_conversations()
        new_chat()
        st.rerun()
    st.markdown("</div>", unsafe_allow_html=True)


conv = active_conv()


def handle_question(question: str):
    if not question:
        return
    if not st.session_state.kb_ready or st.session_state.rag is None:
        conv["messages"].append({"role": "user", "content": question})
        conv["messages"].append({
            "role": "assistant",
            "content": "⚠️ Please upload your documents and click **Process Documents** in the sidebar before asking questions.",
        })
        if conv["title"] == "New Chat":
            conv["title"] = question[:42]
        save_conversations()
        return

    conv["messages"].append({"role": "user", "content": question})
    if conv["title"] == "New Chat":
        conv["title"] = question[:42]

    history = ""
    for m in conv["messages"][:-1]:
        prefix = "User" if m["role"] == "user" else "Assistant"
        history += f"\n{prefix}: {m['content']}"

    with st.spinner("Analyzing documents…"):
        try:
            answer = ask_question(rag=st.session_state.rag, question=question, history=history)
        except Exception:
            logger.exception("Failed to answer question.")
            answer = "Sorry, an error occurred while generating the answer. Please check the logs."

    conv["messages"].append({"role": "assistant", "content": answer})
    save_conversations()


if st.session_state.pending_question:
    q = st.session_state.pending_question
    st.session_state.pending_question = None
    handle_question(q)
    st.rerun()

with st.container(key="main_content_col"):
    st.markdown('<div class="main-wrap">', unsafe_allow_html=True)

    if not conv["messages"]:
        st.markdown(
            f"""
            <div class="landing-wrap">
                <div class="landing-logo">{ongc_logo_img(80)}</div>
                <div class="landing-title">ONGC Enterprise AI</div>
                <div class="landing-sub">Ask questions about your annual reports</div>
            </div>
            """,
            unsafe_allow_html=True,
        )
    else:
        for m in conv["messages"]:
            avatar = "🧑" if m["role"] == "user" else (str(LOGO_PATH) if LOGO_PATH.exists() else "🛢️")
            with st.chat_message(m["role"], avatar=avatar):
                st.markdown(m["content"])

    st.markdown("</div>", unsafe_allow_html=True)

    st.markdown(
        "<div class='disclaimer'>ONGC Enterprise AI may make mistakes. Please verify important information.</div>",
        unsafe_allow_html=True,
    )

user_input = st.chat_input("Ask anything about your annual reports…")
if user_input:
    handle_question(user_input)
    st.rerun()
