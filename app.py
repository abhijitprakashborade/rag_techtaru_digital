"""
app.py
======
Entry point — Streamlit UI for the RAG Chat PDF application.

Startup sequence:
  1. st.set_page_config (MUST be first Streamlit call)
  2. CSS injection
  3. Session state initialisation
  4. Sidebar (upload, document management, export)
  5. Main chat area (history display + streaming input handler)
"""

from __future__ import annotations

import logging
import base64
import os

import streamlit as st
from dotenv import load_dotenv

# ── Bootstrap ──────────────────────────────────────────────
load_dotenv()

# When deployed on Streamlit Community Cloud there is no .env — keys live
# in st.secrets. Bridge them into os.environ before rag.config snapshots
# os.getenv values at import time.
try:
    for _k in ("LLM_PROVIDER", "LLM_MODEL", "GOOGLE_API_KEY",
               "OPENAI_API_KEY", "ANTHROPIC_API_KEY"):
        if not os.getenv(_k):
            _v = st.secrets.get(_k)
            if _v:
                os.environ[_k] = _v
except Exception:  # noqa: BLE001 — no secrets file locally
    pass

from rag.config import (
    APP_ICON,
    APP_SUBTITLE,
    APP_TITLE,
    DUPLICATE_FILE_INFO,
    EMPTY_INPUT_WARNING,
    LARGE_FILE_WARNING,
    LLM_MODEL,
    LLM_PROVIDER,
    NO_DOCS_WARNING,
)
from rag.document_processor import ProcessedPDF, ProcessingError, get_first_page_text, process_pdf
from rag.utils import (
    configure_logging,
    export_chat_to_text,
    format_source_pills,
    get_export_filename,
    human_readable_size,
)
from rag.vector_store import (
    add_documents,
    build_retriever,
    delete_document,
    document_exists,
    get_collection_stats,
    list_indexed_sources,
)
from rag.chain import build_chat_history, build_rag_chain, generate_summary, stream_chain

# ── Logging ────────────────────────────────────────────────
configure_logging()
logger = logging.getLogger(__name__)

# Which API key env var the configured provider requires. Resolved after
# load_dotenv() so LLM_PROVIDER is read from .env when present.
_KEY_BY_PROVIDER: dict[str, str] = {
    "gemini": "GOOGLE_API_KEY",
    "openai": "OPENAI_API_KEY",
    "anthropic": "ANTHROPIC_API_KEY",
}
API_KEY_ENV: str = _KEY_BY_PROVIDER.get(
    os.getenv("LLM_PROVIDER", "gemini").strip().lower(), "GOOGLE_API_KEY"
)

# ── Logo assets ────────────────────────────────────────────
ASSETS_DIR: str = os.path.join(os.path.dirname(os.path.abspath(__file__)), "assets")
LOGO_ICON_PATH: str = os.path.join(ASSETS_DIR, "logo-icon-128.png")
SAMPLE_DOCS_DIR: str = os.path.join(os.path.dirname(os.path.abspath(__file__)), "sample_docs")


@st.cache_data
def load_logo_data_uri() -> str:
    """Return the app logo as an inline base64 data URI.

    Embedding avoids Streamlit's local-file <img> restrictions and makes
    the logo survive caching. Falls back to '' when missing (callers
    then render the emoji mark).
    """
    for name in ("logo-icon.png", "logo-icon-128.png", "logo-256.png"):
        path = os.path.join(ASSETS_DIR, name)
        if os.path.exists(path):
            with open(path, "rb") as fh:
                return "data:image/png;base64," + base64.b64encode(fh.read()).decode()
    return ""

# ──────────────────────────────────────────────────────────────────────────────
# PAGE CONFIG — must be the absolute first Streamlit call
# ──────────────────────────────────────────────────────────────────────────────
_page_icon = LOGO_ICON_PATH if os.path.exists(LOGO_ICON_PATH) else APP_ICON
st.set_page_config(
    page_title=f"{APP_TITLE} · Techtaru Digital",
    page_icon=_page_icon,
    layout="wide",
    initial_sidebar_state="expanded",
    menu_items={
        "About": f"{APP_TITLE} — Chat with your PDF · built for Techtaru Digital",
    },
)

# ──────────────────────────────────────────────────────────────────────────────
# PREMIUM CSS — injected before any st.* call
# ──────────────────────────────────────────────────────────────────────────────
PREMIUM_CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&display=swap');

/* ── Design tokens ─────────────────────────────────────── */
:root {
  --bg:          #07070f;
  --surface:     #0e0e1c;
  --surface-2:   #14142a;
  --surface-3:   #1c1c38;
  --surface-4:   #22224a;
  --accent:      #6366f1;
  --accent-2:    #8b5cf6;
  --accent-glow: rgba(99,102,241,0.35);
  --success:     #10b981;
  --warning:     #f59e0b;
  --danger:      #ef4444;
  --fg:          #f1f5f9;
  --fg-2:        #94a3b8;
  --fg-3:        #475569;
  --border:      rgba(99,102,241,0.14);
  --border-2:    rgba(99,102,241,0.28);
  --radius-sm:   8px;
  --radius-md:   14px;
  --radius-lg:   20px;
  --shadow-glow: 0 0 40px rgba(99,102,241,0.12);
  --transition:  0.2s cubic-bezier(.4,0,.2,1);
}

/* ── Global reset ──────────────────────────────────────── */
*, *::before, *::after { box-sizing: border-box; }

html, body, [data-testid="stApp"] {
  background-color: var(--bg) !important;
  color: var(--fg) !important;
  font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif !important;
}

/* ── Hide Streamlit chrome ───────────────────────────────
   header and stToolbar must stay visible: the floating «/» sidebar
   toggle button is rendered INSIDE the app toolbar — hiding the toolbar
   left the sidebar permanently unrecoverable when collapsed. Both are
   styled transparent below. */
#MainMenu, footer,
.stDeployButton,
[data-testid="stDecoration"],
[data-testid="stStatusWidget"] {
  display: none !important;
}
header,
[data-testid="stToolbar"] {
  background: transparent !important;
  border: none !important;
  box-shadow: none !important;
}
/* The Deploy button also lives in the app toolbar — hide only that one,
   the sidebar toggle (stExpandSidebarButton) must stay clickable. */
[data-testid="stToolbar"] button[data-testid="stBaseButton-header"] {
  display: none !important;
}

/* Sidebar toggle: keep the floating "Expand sidebar" arrow visible and
   clickable while collapsed AND the « collapse button working while
   expanded — full open/close toggle from either side. */
[data-testid="stExpandSidebarButton"] {
  display: flex !important;
  opacity: 1 !important;
  visibility: visible !important;
  z-index: 999 !important;
}

/* ── Ambient background gradient ───────────────────────── */
[data-testid="stApp"]::before {
  content: '';
  position: fixed;
  inset: 0;
  background:
    radial-gradient(ellipse 80% 60% at 20% -10%, rgba(99,102,241,0.18) 0%, transparent 70%),
    radial-gradient(ellipse 60% 50% at 80% 100%, rgba(139,92,246,0.12) 0%, transparent 60%);
  pointer-events: none;
  z-index: 0;
}

/* ── Sidebar ───────────────────────────────────────────── */
[data-testid="stSidebar"] {
  background: var(--surface) !important;
  border-right: 1px solid var(--border) !important;
  backdrop-filter: blur(12px);
}

[data-testid="stSidebar"] > div:first-child {
  padding-top: 1.5rem;
}

/* ── Main content padding ──────────────────────────────── */
.main .block-container {
  padding: 1.5rem 2rem 6rem 2rem;
  max-width: 1100px;
}

/* ── App header ────────────────────────────────────────── */
.app-header {
  display: flex;
  align-items: center;
  gap: 14px;
  margin-bottom: 0.25rem;
  padding-bottom: 1rem;
  border-bottom: 1px solid var(--border);
}
.app-header-icon {
  width: 52px;
  height: 52px;
  border-radius: 14px;
  background: linear-gradient(135deg, rgba(99,102,241,0.14), rgba(139,92,246,0.14));
  display: flex;
  align-items: center;
  justify-content: center;
  font-size: 1.5rem;
  box-shadow: 0 4px 20px rgba(99,102,241,0.25);
  border: 1px solid var(--border);
  flex-shrink: 0;
}
.app-header-icon img { border-radius: 10px; }
.app-header-text h1 {
  margin: 0;
  font-size: 1.55rem;
  font-weight: 800;
  letter-spacing: -0.03em;
  background: linear-gradient(135deg, #fff 0%, #c4b5fd 100%);
  -webkit-background-clip: text;
  -webkit-text-fill-color: transparent;
  background-clip: text;
}
.app-header-text p {
  margin: 0;
  font-size: 0.82rem;
  color: var(--fg-2);
  font-weight: 400;
}

/* ── Status badge row ──────────────────────────────────── */
.status-badges {
  display: flex;
  gap: 8px;
  flex-wrap: wrap;
  margin: 0.75rem 0 1.25rem 0;
}
.badge {
  display: inline-flex;
  align-items: center;
  gap: 5px;
  padding: 4px 12px;
  border-radius: 20px;
  font-size: 11px;
  font-weight: 600;
  letter-spacing: 0.02em;
}
.badge-indigo {
  background: rgba(99,102,241,0.15);
  border: 1px solid rgba(99,102,241,0.3);
  color: #a5b4fc;
}
.badge-green {
  background: rgba(16,185,129,0.12);
  border: 1px solid rgba(16,185,129,0.28);
  color: #6ee7b7;
}
.badge-purple {
  background: rgba(139,92,246,0.12);
  border: 1px solid rgba(139,92,246,0.28);
  color: #c4b5fd;
}

/* ── Chat messages ─────────────────────────────────────── */
[data-testid="stChatMessage"] {
  animation: msgIn 0.28s ease forwards;
  border-radius: var(--radius-md) !important;
  padding: 0.25rem 0 !important;
  background: transparent !important;
  border: none !important;
}

@keyframes msgIn {
  from { opacity: 0; transform: translateY(8px); }
  to   { opacity: 1; transform: translateY(0); }
}

/* User bubble */
[data-testid="stChatMessage"]:has([data-testid="stChatMessageAvatarUser"])
[data-testid="stChatMessageContent"] {
  background: linear-gradient(135deg, #6366f1 0%, #8b5cf6 100%) !important;
  color: #fff !important;
  border-radius: 20px 20px 4px 20px !important;
  max-width: 72% !important;
  margin-left: auto !important;
  padding: 14px 18px !important;
  box-shadow: 0 4px 24px rgba(99,102,241,0.35) !important;
  border: none !important;
}

/* Assistant bubble */
[data-testid="stChatMessage"]:has([data-testid="stChatMessageAvatarAssistant"])
[data-testid="stChatMessageContent"] {
  background: var(--surface-2) !important;
  border: 1px solid var(--border) !important;
  border-radius: 4px 20px 20px 20px !important;
  max-width: 84% !important;
  padding: 14px 18px !important;
  backdrop-filter: blur(8px);
  box-shadow: 0 2px 16px rgba(0,0,0,0.25) !important;
}

/* ── Source citation pills ─────────────────────────────── */
.source-pill {
  display: inline-flex;
  align-items: center;
  gap: 4px;
  background: rgba(99,102,241,0.10);
  border: 1px solid rgba(99,102,241,0.28);
  border-radius: 20px;
  padding: 3px 11px;
  font-size: 11px;
  font-weight: 500;
  color: #a5b4fc;
  margin: 3px 3px 0 0;
  font-family: 'Inter', sans-serif;
  transition: background var(--transition);
}
.source-pill:hover { background: rgba(99,102,241,0.22); }

.sources-container {
  margin-top: 10px;
  padding-top: 10px;
  border-top: 1px solid var(--border);
}
.sources-label {
  font-size: 10px;
  font-weight: 600;
  color: var(--fg-3);
  text-transform: uppercase;
  letter-spacing: 0.08em;
  margin-bottom: 6px;
}

/* ── Searching indicator ───────────────────────────────── */
.searching-indicator {
  display: flex;
  align-items: center;
  gap: 10px;
  padding: 12px 16px;
  background: var(--surface-2);
  border: 1px solid var(--border);
  border-radius: var(--radius-md);
  font-size: 13px;
  color: var(--fg-2);
  margin-bottom: 8px;
}
.dot-pulse {
  position: relative;
  width: 8px;
  height: 8px;
  border-radius: 50%;
  background: var(--accent);
  box-shadow: 0 0 0 0 var(--accent-glow);
  animation: pulse 1.4s infinite;
}
@keyframes pulse {
  0%   { box-shadow: 0 0 0 0 var(--accent-glow); }
  70%  { box-shadow: 0 0 0 10px rgba(99,102,241,0); }
  100% { box-shadow: 0 0 0 0 rgba(99,102,241,0); }
}

/* ── Welcome / empty state ─────────────────────────────── */
.welcome-card {
  text-align: center;
  padding: 3.5rem 2rem;
  background: var(--surface-2);
  border: 1px solid var(--border);
  border-radius: var(--radius-lg);
  margin: 2rem auto;
  max-width: 600px;
  backdrop-filter: blur(10px);
  box-shadow: var(--shadow-glow);
}
.welcome-card .icon { font-size: 3rem; margin-bottom: 1rem; }
.welcome-card h2 {
  font-size: 1.4rem;
  font-weight: 700;
  color: var(--fg);
  margin-bottom: 0.6rem;
}
.welcome-card p { color: var(--fg-2); font-size: 0.9rem; line-height: 1.6; }
.welcome-features {
  display: flex;
  justify-content: center;
  gap: 12px;
  flex-wrap: wrap;
  margin-top: 1.5rem;
}
.feature-chip {
  background: var(--surface-3);
  border: 1px solid var(--border);
  border-radius: 20px;
  padding: 6px 14px;
  font-size: 12px;
  color: var(--fg-2);
  display: flex;
  align-items: center;
  gap: 5px;
}

/* ── Chat input ────────────────────────────────────────── */
[data-testid="stChatInput"] textarea {
  background: var(--surface-2) !important;
  border: 1px solid var(--border) !important;
  border-radius: var(--radius-md) !important;
  color: var(--fg) !important;
  font-family: 'Inter', sans-serif !important;
  font-size: 14px !important;
  transition: border var(--transition), box-shadow var(--transition) !important;
}
[data-testid="stChatInput"] textarea:focus {
  border-color: var(--border-2) !important;
  box-shadow: 0 0 0 3px rgba(99,102,241,0.12) !important;
}

/* ── Sidebar doc card ──────────────────────────────────── */
.doc-card {
  background: var(--surface-2);
  border: 1px solid var(--border);
  border-radius: var(--radius-md);
  padding: 10px 14px;
  margin-bottom: 8px;
  transition: border var(--transition);
}
.doc-card:hover { border-color: var(--border-2); }
.doc-card-name {
  font-size: 13px;
  font-weight: 600;
  color: var(--fg);
  white-space: nowrap;
  overflow: hidden;
  text-overflow: ellipsis;
  max-width: 200px;
}
.doc-card-meta {
  font-size: 11px;
  color: var(--fg-3);
  margin-top: 2px;
}

/* ── Sidebar summary box ───────────────────────────────── */
.summary-box {
  background: linear-gradient(135deg, rgba(99,102,241,0.06), rgba(139,92,246,0.06));
  border: 1px solid var(--border);
  border-radius: var(--radius-md);
  padding: 12px 14px;
  font-size: 12.5px;
  color: var(--fg-2);
  line-height: 1.65;
  margin-top: 8px;
}

/* ── Stat cards row ────────────────────────────────────── */
.stat-row {
  display: flex;
  gap: 8px;
  margin: 0.75rem 0;
}
.stat-card {
  flex: 1;
  background: var(--surface-2);
  border: 1px solid var(--border);
  border-radius: var(--radius-sm);
  padding: 10px;
  text-align: center;
}
.stat-card .stat-val {
  font-size: 1.3rem;
  font-weight: 700;
  color: #a5b4fc;
}
.stat-card .stat-label {
  font-size: 10px;
  color: var(--fg-3);
  text-transform: uppercase;
  letter-spacing: 0.06em;
  margin-top: 2px;
}

/* ── Divider ───────────────────────────────────────────── */
.divider {
  height: 1px;
  background: var(--border);
  margin: 1rem 0;
}

/* ── Streamlit element overrides ───────────────────────── */
[data-testid="stFileUploader"] {
  background: rgba(99,102,241,0.04) !important;
  border: 2px dashed rgba(99,102,241,0.28) !important;
  border-radius: var(--radius-md) !important;
  transition: border var(--transition) !important;
}
[data-testid="stFileUploader"]:hover {
  border-color: rgba(99,102,241,0.55) !important;
}

button[kind="primary"],
.stButton > button {
  background: linear-gradient(135deg, #6366f1, #8b5cf6) !important;
  border: none !important;
  border-radius: var(--radius-sm) !important;
  color: #fff !important;
  font-family: 'Inter', sans-serif !important;
  font-weight: 600 !important;
  transition: opacity var(--transition), box-shadow var(--transition) !important;
}
.stButton > button:hover {
  opacity: 0.88 !important;
  box-shadow: 0 4px 20px rgba(99,102,241,0.4) !important;
}

.stButton > button[kind="secondary"],
button[kind="secondary"] {
  background: var(--surface-3) !important;
  border: 1px solid var(--border) !important;
  color: var(--fg-2) !important;
}

/* ── Progress bar ──────────────────────────────────────── */
.stProgress > div > div > div > div {
  background: linear-gradient(90deg, #6366f1, #8b5cf6) !important;
}

/* ── Alerts ────────────────────────────────────────────── */
[data-testid="stAlert"] {
  border-radius: var(--radius-md) !important;
  font-family: 'Inter', sans-serif !important;
}

/* ── Scrollbar ─────────────────────────────────────────── */
::-webkit-scrollbar { width: 4px; height: 4px; }
::-webkit-scrollbar-thumb {
  background: rgba(99,102,241,0.4);
  border-radius: 2px;
}
::-webkit-scrollbar-track { background: transparent; }

/* ── Sidebar section headings ──────────────────────────── */
.sidebar-section-title {
  font-size: 10px;
  font-weight: 700;
  color: var(--fg-3);
  text-transform: uppercase;
  letter-spacing: 0.1em;
  margin: 1.25rem 0 0.5rem 0;
}
</style>
"""

# ──────────────────────────────────────────────────────────────────────────────
# SESSION STATE INITIALISATION
# ──────────────────────────────────────────────────────────────────────────────
def init_session_state() -> None:
    """Initialise all st.session_state keys with safe defaults.

    Called once at the top of app.py before any rendering.
    Prevents KeyError on first run and after page refresh.
    """
    defaults: dict[str, object] = {
        "messages": [],           # list[dict[str, str]] — chat history
        "uploaded_docs": {},      # dict[sha256, {filename, size, page_count, chunks, summary}]
        "rag_chain": None,        # built LCEL chain or None
        "processing": False,      # True while embedding is running
    }
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value


# ──────────────────────────────────────────────────────────────────────────────
# CHAIN CACHE — rebuild when documents change
# ──────────────────────────────────────────────────────────────────────────────
def get_or_build_chain(contextualize: bool = True):
    """Return the cached RAG chain, rebuilding if the retriever may be stale.

    Args:
        contextualize: False when there is no chat history yet — skips the
            extra LLM call that reformulates the question (fast path).

    Returns:
        LCEL Runnable ready for .stream() calls, or None if no docs indexed.
    """
    sources = list_indexed_sources()
    if not sources:
        return None
    retriever = build_retriever()
    return build_rag_chain(retriever, contextualize=contextualize)


def api_key_status() -> tuple[bool, str]:
    """Validate the provider API key.

    Returns:
        (ok, message) — ok is True when a real, non-placeholder key is set.
    """
    key = (os.getenv(API_KEY_ENV) or "").strip()
    if not key:
        return False, f"⚠️ `{API_KEY_ENV}` not set in `.env`"
    lowered = key.lower()
    if lowered.startswith("your-") or "here" in lowered:
        return False, f"⚠️ `{API_KEY_ENV}` still contains the placeholder value — put your real key in `.env`"
    return True, ""


# ──────────────────────────────────────────────────────────────────────────────
# SIDEBAR
# ──────────────────────────────────────────────────────────────────────────────
def render_sidebar() -> None:
    """Render the full sidebar: upload panel, document list, stats, export."""
    with st.sidebar:
        # ── Sidebar header ───────────────────────────────
        logo_uri = load_logo_data_uri()
        logo_html = (
            f'<img src="{logo_uri}" width="34" style="border-radius:8px;">'
            if logo_uri
            else "✨"
        )
        st.markdown(
            f"""
            <div style="display:flex;align-items:center;gap:10px;margin-bottom:1rem;">
              <div style="width:38px;height:38px;border-radius:10px;
                          background:linear-gradient(135deg,rgba(99,102,241,.15),rgba(139,92,246,.15));
                          display:flex;align-items:center;justify-content:center;
                          font-size:1.1rem;box-shadow:0 3px 12px rgba(99,102,241,0.25);">
                {logo_html}
              </div>
              <div>
                <div style="font-weight:700;font-size:0.95rem;color:#f1f5f9;">{APP_TITLE}</div>
                <div style="font-size:10px;color:#475569;margin-top:1px;">Techtaru Digital</div>
              </div>
            </div>
            """,
            unsafe_allow_html=True,
        )
        st.markdown('<div class="divider"></div>', unsafe_allow_html=True)

        # ── Upload section ───────────────────────────────
        st.markdown('<p class="sidebar-section-title">📂 Upload Documents</p>', unsafe_allow_html=True)
        uploaded_files = st.file_uploader(
            "Upload PDF(s)",
            type=["pdf"],
            accept_multiple_files=True,
            help="Upload one or more PDF documents to chat with.",
            label_visibility="collapsed",
        )

        if uploaded_files:
            _handle_uploads(uploaded_files)

        # ── Document list ────────────────────────────────
        indexed = list_indexed_sources()
        if indexed:
            st.markdown('<p class="sidebar-section-title">📋 Indexed Documents</p>', unsafe_allow_html=True)
            _render_document_list(indexed)

        # ── Stats row ────────────────────────────────────
        stats = get_collection_stats()
        if stats["total_chunks"] > 0:
            st.markdown(
                f"""
                <div class="stat-row">
                  <div class="stat-card">
                    <div class="stat-val">{stats['unique_sources']}</div>
                    <div class="stat-label">Docs</div>
                  </div>
                  <div class="stat-card">
                    <div class="stat-val">{stats['total_chunks']}</div>
                    <div class="stat-label">Chunks</div>
                  </div>
                </div>
                """,
                unsafe_allow_html=True,
            )

        st.markdown('<div class="divider"></div>', unsafe_allow_html=True)

        # ── Export section ───────────────────────────────
        st.markdown('<p class="sidebar-section-title">💾 Export</p>', unsafe_allow_html=True)
        if st.session_state.messages:
            filenames = [
                meta["filename"]
                for meta in st.session_state.uploaded_docs.values()
            ]
            export_text = export_chat_to_text(st.session_state.messages, filenames)
            st.download_button(
                label="⬇️ Download Chat (.txt)",
                data=export_text,
                file_name=get_export_filename(),
                mime="text/plain",
                use_container_width=True,
            )
        else:
            st.caption("Start chatting to enable export.")

        # ── Clear chat ───────────────────────────────────
        st.markdown('<div class="divider"></div>', unsafe_allow_html=True)
        if st.button("🗑️ Clear Chat History", use_container_width=True, type="secondary"):
            st.session_state.messages = []
            st.rerun()

        # ── API key check ────────────────────────────────
        key_ok, key_msg = api_key_status()
        if not key_ok:
            st.error(key_msg)

        st.markdown(
            '<div style="font-size:10px;color:#334155;margin-top:2rem;text-align:center;">'
            f"{LLM_MODEL} · LangChain LCEL · ChromaDB"
            "</div>",
            unsafe_allow_html=True,
        )


def _handle_uploads(uploaded_files: list) -> None:
    """Process each uploaded file: dedup → extract → chunk → embed → summarise."""
    for uploaded_file in uploaded_files:
        file_bytes = uploaded_file.read()
        filename = uploaded_file.name
        file_size = len(file_bytes)

        # ── SHA-256 dedup check ──────────────────────────
        from rag.document_processor import compute_sha256
        sha256 = compute_sha256(file_bytes)

        if document_exists(sha256):
            st.info(DUPLICATE_FILE_INFO.format(filename=filename))
            logger.info("Duplicate detected: '%s' (sha256=%s…)", filename, sha256[:12])
            continue

        # ── Large file warning ───────────────────────────
        file_size_mb = file_size / (1024 * 1024)
        if file_size_mb > 15:
            st.warning(LARGE_FILE_WARNING)

        # ── Process PDF ──────────────────────────────────
        with st.spinner(f"📄 Processing **{filename}**…"):
            result = process_pdf(file_bytes, filename)

        if isinstance(result, ProcessingError):
            st.error(result.message)
            logger.error("Processing error for '%s': %s", filename, result.message)
            continue

        assert isinstance(result, ProcessedPDF)

        # ── Embed with progress ──────────────────────────
        progress_bar = st.progress(0.0, text=f"Embedding {filename}…")
        try:
            stored_count = add_documents(result.chunks, progress_callback=progress_bar)
        except Exception as exc:  # noqa: BLE001
            st.error(f"❌ Embedding failed for **{filename}**: {exc}")
            logger.error("Embedding failed for '%s': %s", filename, exc)
            progress_bar.empty()
            continue

        progress_bar.empty()
        st.success(
            f"✅ **{filename}** indexed — {result.page_count} pages · {stored_count} chunks"
        )

        # ── Generate summary ─────────────────────────────
        summary_text = ""
        try:
            first_text = get_first_page_text(file_bytes, filename)
            if first_text:
                with st.spinner("🧠 Generating summary…"):
                    summary_text = generate_summary(first_text)
        except Exception as exc:  # noqa: BLE001
            logger.warning("Summary generation failed for '%s': %s", filename, exc)
            summary_text = "_Summary unavailable._"

        # ── Store in session ─────────────────────────────
        st.session_state.uploaded_docs[sha256] = {
            "filename": filename,
            "size": file_size,
            "page_count": result.page_count,
            "chunk_count": stored_count,
            "summary": summary_text,
        }

        if summary_text:
            with st.expander(f"📝 Summary: {filename}", expanded=True):
                st.markdown(
                    f'<div class="summary-box">{summary_text}</div>',
                    unsafe_allow_html=True,
                )


class _SampleFile:
    """Duck-typed stand-in for st.UploadedFile, backed by a bundled sample PDF."""

    def __init__(self, path: str):
        self.name = os.path.basename(path)
        with open(path, "rb") as fh:
            self._data = fh.read()

    def read(self) -> bytes:
        return self._data


def load_sample_documents() -> None:
    """Index every PDF bundled in sample_docs/ (first-run demo path)."""
    pdfs = sorted(
        os.path.join(SAMPLE_DOCS_DIR, name)
        for name in os.listdir(SAMPLE_DOCS_DIR)
        if name.lower().endswith(".pdf")
    ) if os.path.isdir(SAMPLE_DOCS_DIR) else []
    if pdfs:
        _handle_uploads([_SampleFile(path) for path in pdfs])


def render_get_started() -> None:
    """First-run onboarding rendered in the MAIN area.

    Shows even if the sidebar is collapsed: an upload panel plus a
    one-click "load the bundled sample PDF" button, so a first-time
    visitor can reach a working demo in a single click.
    """
    with st.container(border=True):
        st.markdown("### 📂 Get started")
        ups = st.file_uploader(
            "Upload PDF(s)",
            type=["pdf"],
            accept_multiple_files=True,
            key="main_uploader",
            label_visibility="collapsed",
        )
        if ups:
            _handle_uploads(ups)
        st.markdown('<div class="divider"></div>', unsafe_allow_html=True)
        col_btn, col_hint = st.columns([1, 2])
        with col_btn:
            if st.button("⚡ Load sample PDF", use_container_width=True, type="primary"):
                load_sample_documents()
                st.rerun()
        with col_hint:
            st.caption("No PDF handy? Load the bundled sample document and ask it anything.")


def _render_document_list(indexed_sources: list[str]) -> None:
    """Render each indexed document as a card with a delete button."""
    # Build a sha256 → metadata reverse lookup
    sha_to_meta = {
        meta["filename"]: (sha, meta)
        for sha, meta in st.session_state.uploaded_docs.items()
    }

    for source in indexed_sources:
        col_name, col_del = st.columns([4, 1])

        sha, meta = sha_to_meta.get(source, (None, {}))
        size_label = human_readable_size(meta.get("size", 0)) if meta else ""
        pages_label = f"{meta.get('page_count', '?')} pages" if meta else ""

        with col_name:
            st.markdown(
                f"""
                <div class="doc-card">
                  <div class="doc-card-name" title="{source}">📄 {source}</div>
                  <div class="doc-card-meta">{size_label} · {pages_label}</div>
                </div>
                """,
                unsafe_allow_html=True,
            )

        with col_del:
            if st.button("🗑️", key=f"del_{source}", help=f"Remove {source}"):
                deleted = delete_document(source)
                if sha and sha in st.session_state.uploaded_docs:
                    del st.session_state.uploaded_docs[sha]
                st.success(f"Removed **{source}** ({deleted} chunks)")
                logger.info("User deleted '%s' — %d chunks removed.", source, deleted)
                st.rerun()


# ──────────────────────────────────────────────────────────────────────────────
# MAIN CHAT AREA
# ──────────────────────────────────────────────────────────────────────────────
def render_header() -> None:
    """Render the main area top header with badges."""
    indexed_count = len(list_indexed_sources())
    api_ok, _ = api_key_status()

    logo_uri = load_logo_data_uri()
    header_logo_html = (
        f'<img src="{logo_uri}" width="40" style="border-radius:10px;box-shadow:0 4px 20px rgba(99,102,241,0.35);">'
        if logo_uri
        else "✨"
    )

    st.markdown(
        f"""
        <div class="app-header">
          <div class="app-header-icon">{header_logo_html}</div>
          <div class="app-header-text">
            <h1>{APP_TITLE}</h1>
            <p>{APP_SUBTITLE}</p>
          </div>
        </div>
        <div class="status-badges">
          <span class="badge badge-indigo">⚡ {LLM_MODEL}</span>
          <span class="badge badge-purple">🗄️ ChromaDB Persistent</span>
          <span class="badge badge-indigo">🔍 MMR Retrieval</span>
          {'<span class="badge badge-green">🔑 API Key ✓</span>' if api_ok else
           '<span class="badge" style="background:rgba(239,68,68,0.12);border:1px solid rgba(239,68,68,0.3);color:#fca5a5;">❌ No API Key</span>'}
          {'<span class="badge badge-green">📄 ' + str(indexed_count) + ' doc(s) indexed</span>' if indexed_count > 0 else ''}
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_welcome() -> None:
    """Render the welcome card when no messages exist."""
    st.markdown(
        """
        <div class="welcome-card">
          <div class="icon">🤖</div>
          <h2>Ready to Analyse Your Documents</h2>
          <p>
            Upload one or more PDFs in the sidebar, then ask anything about them.
            I'll retrieve the most relevant passages and answer strictly from your documents —
            with exact page citations.
          </p>
          <div class="welcome-features">
            <span class="feature-chip">📄 Multi-PDF support</span>
            <span class="feature-chip">🔍 MMR retrieval</span>
            <span class="feature-chip">🧠 Chat memory</span>
            <span class="feature-chip">⚡ Streaming answers</span>
            <span class="feature-chip">📎 Page citations</span>
            <span class="feature-chip">💾 Export chat</span>
          </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_chat_history() -> None:
    """Render all previous messages from session state."""
    for msg in st.session_state.messages:
        avatar = "🧑" if msg["role"] == "user" else "🤖"
        with st.chat_message(msg["role"], avatar=avatar):
            st.markdown(msg["content"], unsafe_allow_html=True)
            # Re-render source pills if stored
            if msg["role"] == "assistant" and msg.get("sources_html"):
                st.markdown(
                    f'<div class="sources-container">'
                    f'<div class="sources-label">Sources</div>'
                    f'{msg["sources_html"]}'
                    f"</div>",
                    unsafe_allow_html=True,
                )


def handle_chat_input() -> None:
    """Process user input: validate → retrieve → stream → display citations."""
    user_input: str | None = st.chat_input(
        "Ask anything about your uploaded documents…",
        key="chat_input_main",
    )

    if not user_input:
        return

    # ── Guard: empty / whitespace ────────────────────────
    if not user_input.strip():
        st.warning(EMPTY_INPUT_WARNING)
        return

    # ── Guard: no documents indexed ──────────────────────
    indexed_sources = list_indexed_sources()
    if not indexed_sources:
        st.warning(NO_DOCS_WARNING)
        return

    # ── Render user bubble ───────────────────────────────
    with st.chat_message("user", avatar="🧑"):
        st.markdown(user_input)

    st.session_state.messages.append({"role": "user", "content": user_input})

    # ── Searching indicator ──────────────────────────────
    search_placeholder = st.empty()
    search_placeholder.markdown(
        '<div class="searching-indicator">'
        '<div class="dot-pulse"></div>'
        "🔍 Searching documents…"
        "</div>",
        unsafe_allow_html=True,
    )

    # ── Build / reuse chain ──────────────────────────────
    chat_history = build_chat_history(st.session_state.messages[:-1])
    chain = get_or_build_chain(contextualize=bool(chat_history))
    if chain is None:
        search_placeholder.empty()
        st.warning(NO_DOCS_WARNING)
        return

    # ── Stream response ──────────────────────────────────
    search_placeholder.empty()

    full_answer = ""
    context_docs: list = []

    with st.chat_message("assistant", avatar="🤖"):
        answer_placeholder = st.empty()

        try:
            # One pass: stream the answer while capturing the retrieved
            # context documents emitted by the chain (used for citations).
            for token in stream_chain(chain, user_input.strip(), chat_history, context_docs):
                full_answer += token
                answer_placeholder.markdown(full_answer + "▌")

            answer_placeholder.markdown(full_answer)

        except Exception as exc:  # noqa: BLE001
            logger.error("Chain invocation failed: %s", exc, exc_info=True)
            exc_text = str(exc)
            looks_like_auth = (
                "401" in exc_text or "403" in exc_text or "api key" in exc_text.lower()
            )
            if looks_like_auth:
                err_msg = (
                    f"⚠️ **Authentication failed.** Your {API_KEY_ENV} is invalid or "
                    f"not configured for the {LLM_PROVIDER} provider.\n\n"
                    f"```\n{type(exc).__name__}\n```\n\n"
                    f"Open `.env`, set `{API_KEY_ENV}` to a valid key, then restart the app."
                )
            else:
                err_msg = (
                    f"⚠️ **An error occurred while generating the response.**\n\n"
                    f"```\n{type(exc).__name__}: {exc}\n```\n\n"
                    "Please try again. If the issue persists, check your API key "
                    "and rate limits."
                )
            answer_placeholder.markdown(err_msg)
            st.session_state.messages.append(
                {"role": "assistant", "content": err_msg, "sources_html": ""}
            )
            return

        # ── Citations from the captured context (no second retrieval) ──
        sources_html = ""
        if context_docs:
            sources_html = format_source_pills(context_docs)
            if sources_html:
                st.markdown(
                    f'<div class="sources-container">'
                    f'<div class="sources-label">Sources</div>'
                    f"{sources_html}"
                    f"</div>",
                    unsafe_allow_html=True,
                )

    # ── Persist assistant message ────────────────────────
    st.session_state.messages.append(
        {
            "role": "assistant",
            "content": full_answer,
            "sources_html": sources_html,
        }
    )


# ──────────────────────────────────────────────────────────────────────────────
# ENTRY POINT
# ──────────────────────────────────────────────────────────────────────────────
def main() -> None:
    """Application entry point.

    Sequence: inject CSS → init state → sidebar → header → chat.
    """
    # 1. Inject premium CSS
    st.markdown(PREMIUM_CSS, unsafe_allow_html=True)

    # 2. Initialise session state
    init_session_state()

    # 3. Sidebar
    render_sidebar()

    # 4. Main area header
    render_header()

    # 5. Chat history or welcome card
    if not st.session_state.messages:
        render_welcome()
    else:
        render_chat_history()

    # 5b. First-run onboarding (visible even with the sidebar collapsed)
    if not list_indexed_sources():
        render_get_started()

    # 6. Chat input handler (always present)
    handle_chat_input()


if __name__ == "__main__":
    main()
