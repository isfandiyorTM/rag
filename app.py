"""
Smart FAQ Chatbot — Retrieval-Augmented Generation (RAG)
Business Intelligence Midterm Project
"""

from __future__ import annotations

import os
import time
from datetime import datetime
from typing import List, Dict

import streamlit as st

# ── Page config ────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="FAQ Assistant",
    page_icon="💬",
    layout="wide",
    initial_sidebar_state="expanded",
)

from utils.document_processor import process_document
from utils.vector_store import VectorStore
from utils.rag_pipeline import get_embeddings, query as rag_query


# ── Design tokens ──────────────────────────────────────────────────────────────
CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600&display=swap');

/* ── Reset / Base ─────────────────────────────────────────── */
*, *::before, *::after { box-sizing: border-box; margin: 0; }

html, body, [class*="css"] {
  font-family: 'Inter', system-ui, -apple-system, sans-serif !important;
  -webkit-font-smoothing: antialiased;
}

/* Hide Streamlit chrome */
#MainMenu, footer, [data-testid="stToolbar"] { display: none !important; }
header[data-testid="stHeader"] { display: none !important; }

/* ── Backgrounds ──────────────────────────────────────────── */
.stApp,
[data-testid="stAppViewContainer"],
[data-testid="stMain"] {
  background: #0f172a !important;
}

[data-testid="stSidebar"] {
  background: #0c1322 !important;
  border-right: 1px solid rgba(148,163,184,0.07) !important;
}
[data-testid="stSidebar"] > div:first-child {
  padding: 24px 18px 24px !important;
}

/* bottom bar (chat input area) */
[data-testid="stBottom"],
[data-testid="stBottom"] > div {
  background: #0f172a !important;
  border-top: 1px solid rgba(148,163,184,0.07) !important;
  padding-top: 10px !important;
}

/* ── Scrollbar ────────────────────────────────────────────── */
::-webkit-scrollbar { width: 4px; height: 4px; }
::-webkit-scrollbar-track { background: transparent; }
::-webkit-scrollbar-thumb { background: rgba(124,92,252,0.2); border-radius: 4px; }
::-webkit-scrollbar-thumb:hover { background: rgba(124,92,252,0.4); }

/* ── Divider ──────────────────────────────────────────────── */
hr {
  border: none !important;
  border-top: 1px solid rgba(148,163,184,0.08) !important;
  margin: 20px 0 !important;
}

/* ── Buttons ──────────────────────────────────────────────── */
.stButton > button {
  border-radius: 8px !important;
  font-family: 'Inter', sans-serif !important;
  font-size: 0.78rem !important;
  font-weight: 500 !important;
  height: 34px !important;
  transition: all 0.15s ease !important;
  border: 1px solid rgba(148,163,184,0.12) !important;
  background: rgba(255,255,255,0.03) !important;
  color: #94a3b8 !important;
  letter-spacing: 0.01em !important;
}
.stButton > button:hover {
  background: rgba(255,255,255,0.07) !important;
  color: #e2e8f0 !important;
  border-color: rgba(148,163,184,0.2) !important;
  transform: none !important;
}
.stButton > button[kind="primary"] {
  background: linear-gradient(135deg, #7c5cfc 0%, #22d3ee 100%) !important;
  border: none !important;
  color: #fff !important;
  font-weight: 600 !important;
}
.stButton > button[kind="primary"]:hover {
  opacity: 0.88 !important;
  box-shadow: 0 4px 20px rgba(124,92,252,0.35) !important;
}

/* ── Text inputs ──────────────────────────────────────────── */
.stTextInput > div > div > input,
.stTextArea textarea {
  background: rgba(255,255,255,0.03) !important;
  border: 1px solid rgba(148,163,184,0.1) !important;
  border-radius: 8px !important;
  color: #f1f5f9 !important;
  font-family: 'Inter', sans-serif !important;
}
.stTextInput > div > div > input::placeholder { color: #475569 !important; }
.stTextInput > div > div > input:focus,
.stTextArea textarea:focus {
  border-color: rgba(124,92,252,0.45) !important;
  box-shadow: 0 0 0 3px rgba(124,92,252,0.1) !important;
}
.stTextInput label, .stTextArea label { color: #64748b !important; font-size: 0.75rem !important; }

/* ── Chat input ───────────────────────────────────────────── */
[data-testid="stChatInput"] {
  background: rgba(30,41,59,0.8) !important;
  border: 1px solid rgba(148,163,184,0.12) !important;
  border-radius: 12px !important;
  backdrop-filter: blur(8px) !important;
}
[data-testid="stChatInput"]:focus-within {
  border-color: rgba(124,92,252,0.4) !important;
  box-shadow: 0 0 0 3px rgba(124,92,252,0.08) !important;
}
[data-testid="stChatInput"] textarea {
  color: #f1f5f9 !important;
  background: transparent !important;
  font-family: 'Inter', sans-serif !important;
  font-size: 0.88rem !important;
}
[data-testid="stChatInput"] textarea::placeholder { color: #475569 !important; }
[data-testid="stChatInputSubmitButton"] svg { fill: #7c5cfc !important; }

/* ── Chat messages ────────────────────────────────────────── */
[data-testid="stChatMessage"] {
  padding: 4px 0 !important;
  gap: 12px !important;
  background: transparent !important;
  border: none !important;
}
[data-testid="stChatMessageContent"] {
  background: rgba(30,41,59,0.55) !important;
  border: 1px solid rgba(148,163,184,0.09) !important;
  border-radius: 4px 14px 14px 14px !important;
  padding: 12px 16px !important;
  color: #e2e8f0 !important;
  font-size: 0.88rem !important;
  line-height: 1.7 !important;
  backdrop-filter: blur(8px) !important;
  max-width: 82% !important;
}

/* User message — flip layout & bubble color */
[data-testid="stChatMessage"]:has([data-testid="stChatMessageAvatarUser"]) {
  flex-direction: row-reverse !important;
}
[data-testid="stChatMessage"]:has([data-testid="stChatMessageAvatarUser"]) [data-testid="stChatMessageContent"] {
  background: rgba(124,92,252,0.13) !important;
  border-color: rgba(124,92,252,0.25) !important;
  border-radius: 14px 4px 14px 14px !important;
  color: #e2e8f0 !important;
}

/* Avatars */
[data-testid="stChatMessageAvatarUser"],
[data-testid="stChatMessageAvatarAssistant"] {
  border-radius: 8px !important;
  width: 32px !important;
  height: 32px !important;
  flex-shrink: 0 !important;
}
[data-testid="stChatMessageAvatarAssistant"] {
  background: linear-gradient(135deg, #7c5cfc 0%, #22d3ee 100%) !important;
  border: none !important;
}
[data-testid="stChatMessageAvatarUser"] {
  background: rgba(124,92,252,0.2) !important;
  border: 1px solid rgba(124,92,252,0.3) !important;
}

/* markdown inside messages */
[data-testid="stChatMessageContent"] p { margin: 0 0 6px !important; }
[data-testid="stChatMessageContent"] p:last-child { margin-bottom: 0 !important; }
[data-testid="stChatMessageContent"] ul, [data-testid="stChatMessageContent"] ol {
  padding-left: 18px !important; margin: 6px 0 !important;
}
[data-testid="stChatMessageContent"] li { margin: 3px 0 !important; }
[data-testid="stChatMessageContent"] strong { color: #f1f5f9 !important; font-weight: 600 !important; }
[data-testid="stChatMessageContent"] code {
  background: rgba(124,92,252,0.15) !important;
  color: #a78bfa !important;
  border-radius: 4px !important;
  padding: 1px 5px !important;
  font-size: 0.82rem !important;
}

/* ── File uploader ────────────────────────────────────────── */
[data-testid="stFileUploaderDropzone"] {
  background: rgba(255,255,255,0.02) !important;
  border: 1px dashed rgba(124,92,252,0.22) !important;
  border-radius: 10px !important;
  transition: all 0.2s ease !important;
}
[data-testid="stFileUploaderDropzone"]:hover {
  border-color: rgba(124,92,252,0.5) !important;
  background: rgba(124,92,252,0.04) !important;
}
[data-testid="stFileUploaderDropzoneInstructions"] { color: #475569 !important; font-size: 0.8rem !important; }

/* ── Progress bar ─────────────────────────────────────────── */
[data-testid="stProgressBar"] > div > div {
  background: linear-gradient(90deg, #7c5cfc, #22d3ee) !important;
  border-radius: 4px !important;
}

/* ── Expander ─────────────────────────────────────────────── */
[data-testid="stExpander"] {
  background: rgba(255,255,255,0.02) !important;
  border: 1px solid rgba(148,163,184,0.08) !important;
  border-radius: 10px !important;
}
[data-testid="stExpander"] summary {
  color: #64748b !important;
  font-size: 0.78rem !important;
  font-weight: 500 !important;
}
[data-testid="stExpander"] summary:hover { color: #94a3b8 !important; }

/* ── Alerts ───────────────────────────────────────────────── */
[data-testid="stAlert"] { border-radius: 10px !important; border-width: 1px !important; }

/* ── Spinner ──────────────────────────────────────────────── */
[data-testid="stSpinner"] > div { border-top-color: #7c5cfc !important; }

/* ── Metrics ──────────────────────────────────────────────── */
[data-testid="stMetricValue"] { color: #7c5cfc !important; font-weight: 600 !important; }
[data-testid="stMetricLabel"] { color: #475569 !important; font-size: 0.7rem !important; letter-spacing: 0.04em !important; }
[data-testid="stMetric"] { padding: 10px 14px !important; }

/* ── Captions ─────────────────────────────────────────────── */
.stCaption { color: #475569 !important; font-size: 0.72rem !important; }
</style>
"""
st.markdown(CSS, unsafe_allow_html=True)


# ── API key ────────────────────────────────────────────────────────────────────
def _load_api_key() -> str:
    try:
        return st.secrets["GROQ_API_KEY"]
    except Exception:
        return os.environ.get("GROQ_API_KEY", "")

GROQ_API_KEY = _load_api_key()


# ── Session state ──────────────────────────────────────────────────────────────
def _init():
    defaults = {
        "messages": [],
        "vector_store": VectorStore(),
        "indexed_docs": [],
        "conversation_log": [],
        "pending_question": "",
    }
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v

_init()


# ── Helpers ────────────────────────────────────────────────────────────────────
def _ts() -> str:
    return datetime.now().strftime("%H:%M")


def _ingest_file(file) -> int:
    file_bytes = file.read()
    chunks = process_document(file_bytes, file.name)
    texts = [c["text"] for c in chunks]
    embeddings = get_embeddings(texts)
    st.session_state.vector_store.add(chunks, embeddings)
    return len(chunks)


# ── Sidebar ────────────────────────────────────────────────────────────────────
with st.sidebar:

    # Logo
    st.markdown("""
<div style="display:flex;align-items:center;gap:10px;margin-bottom:24px;">
  <div style="
    width:36px;height:36px;border-radius:10px;
    background:linear-gradient(135deg,#7c5cfc,#22d3ee);
    display:flex;align-items:center;justify-content:center;
    font-size:1rem;flex-shrink:0;">💬</div>
  <div>
    <div style="color:#f1f5f9;font-weight:600;font-size:0.95rem;letter-spacing:-0.01em;">FAQ Assistant</div>
    <div style="color:#475569;font-size:0.7rem;margin-top:1px;">Powered by Groq · RAG</div>
  </div>
</div>
""", unsafe_allow_html=True)

    # ── KB stats ───────────────────────────────────────────────────────────────
    kb: VectorStore = st.session_state.vector_store

    col_a, col_b = st.columns(2)
    with col_a:
        st.markdown(f"""
<div style="background:rgba(124,92,252,0.07);border:1px solid rgba(124,92,252,0.15);
border-radius:10px;padding:10px 14px;">
  <div style="color:#7c5cfc;font-size:1.3rem;font-weight:700;letter-spacing:-0.02em;">{kb.num_chunks}</div>
  <div style="color:#475569;font-size:0.68rem;margin-top:2px;text-transform:uppercase;letter-spacing:0.06em;">Chunks</div>
</div>""", unsafe_allow_html=True)
    with col_b:
        st.markdown(f"""
<div style="background:rgba(34,211,238,0.07);border:1px solid rgba(34,211,238,0.15);
border-radius:10px;padding:10px 14px;">
  <div style="color:#22d3ee;font-size:1.3rem;font-weight:700;letter-spacing:-0.02em;">{len(kb.sources)}</div>
  <div style="color:#475569;font-size:0.68rem;margin-top:2px;text-transform:uppercase;letter-spacing:0.06em;">Docs</div>
</div>""", unsafe_allow_html=True)

    st.markdown("<div style='margin-top:16px'></div>", unsafe_allow_html=True)

    # ── Knowledge Base ─────────────────────────────────────────────────────────
    st.markdown("""<div style="color:#334155;font-size:0.68rem;font-weight:600;
letter-spacing:0.1em;text-transform:uppercase;margin-bottom:10px;">Knowledge Base</div>""",
                unsafe_allow_html=True)

    uploaded_files = st.file_uploader(
        "drop files here",
        type=["pdf", "txt", "docx"],
        accept_multiple_files=True,
        label_visibility="collapsed",
    )

    s_col, u_col = st.columns(2)
    with s_col:
        load_sample = st.button("Load Sample", use_container_width=True)
    with u_col:
        process_btn = st.button(
            "Process",
            use_container_width=True,
            disabled=not uploaded_files,
            type="primary",
        )

    if load_sample:
        with st.spinner("Indexing…"):
            try:
                with open("data/sample_faq.txt", "rb") as f:
                    raw = f.read()
                kb.remove_source("sample_faq.txt")
                chunks = process_document(raw, "sample_faq.txt")
                embs = get_embeddings([c["text"] for c in chunks])
                kb.add(chunks, embs)
                if "sample_faq.txt" not in st.session_state.indexed_docs:
                    st.session_state.indexed_docs.append("sample_faq.txt")
                st.toast(f"Loaded {len(chunks)} chunks", icon="✅")
                st.rerun()
            except Exception as e:
                st.error(str(e))

    if process_btn and uploaded_files:
        bar = st.progress(0, text="")
        total = 0
        for i, uf in enumerate(uploaded_files):
            bar.progress(i / len(uploaded_files), text=f"{uf.name}")
            try:
                n = _ingest_file(uf)
                total += n
                if uf.name not in st.session_state.indexed_docs:
                    st.session_state.indexed_docs.append(uf.name)
            except Exception as e:
                st.error(f"{uf.name}: {e}")
        bar.progress(1.0, text="Done")
        time.sleep(0.6)
        bar.empty()
        st.toast(f"{total} chunks from {len(uploaded_files)} file(s)", icon="✅")
        st.rerun()

    # Indexed doc list
    if kb.sources:
        st.markdown("<div style='margin-top:12px'></div>", unsafe_allow_html=True)
        for src in kb.sources:
            c1, c2 = st.columns([5, 1])
            with c1:
                st.markdown(f"""<div style="color:#64748b;font-size:0.75rem;
padding:4px 0;white-space:nowrap;overflow:hidden;text-overflow:ellipsis;">
📄 {src}</div>""", unsafe_allow_html=True)
            with c2:
                if st.button("✕", key=f"rm_{src}", help=f"Remove {src}"):
                    kb.remove_source(src)
                    if src in st.session_state.indexed_docs:
                        st.session_state.indexed_docs.remove(src)
                    st.rerun()

    st.divider()

    # ── Conversation history ───────────────────────────────────────────────────
    st.markdown("""<div style="color:#334155;font-size:0.68rem;font-weight:600;
letter-spacing:0.1em;text-transform:uppercase;margin-bottom:10px;">History</div>""",
                unsafe_allow_html=True)

    log = st.session_state.conversation_log
    if log:
        for entry in reversed(log[-12:]):
            st.markdown(f"""
<div style="padding:7px 10px;border-radius:8px;background:rgba(255,255,255,0.02);
border:1px solid rgba(148,163,184,0.07);margin-bottom:5px;">
  <div style="color:#94a3b8;font-size:0.78rem;white-space:nowrap;
overflow:hidden;text-overflow:ellipsis;">
    {entry['question'][:48]}{'…' if len(entry['question']) > 48 else ''}
  </div>
  <div style="color:#334155;font-size:0.65rem;margin-top:3px;">{entry['timestamp']}</div>
</div>""", unsafe_allow_html=True)

        if st.button("Clear chat", use_container_width=True):
            st.session_state.messages = []
            st.session_state.conversation_log = []
            st.rerun()
    else:
        st.markdown("""<div style="color:#334155;font-size:0.78rem;
padding:8px 0;">No conversations yet.</div>""", unsafe_allow_html=True)


# ── Guard ──────────────────────────────────────────────────────────────────────
if not GROQ_API_KEY:
    st.error(
        "**`GROQ_API_KEY` is not configured.** "
        "Add it under App Settings → Secrets on Streamlit Cloud, "
        "or set the `GROQ_API_KEY` environment variable locally.",
        icon="🔑",
    )
    st.stop()


# ── Main area ──────────────────────────────────────────────────────────────────
kb_ready = kb.num_chunks > 0

# Header
status_dot = (
    '<span style="color:#22d3ee">●</span> Ready'
    if kb_ready else
    '<span style="color:#334155">●</span> No docs'
)
st.markdown(f"""
<div style="
  display:flex;align-items:center;justify-content:space-between;
  padding:0 4px;margin-bottom:28px;">
  <div>
    <h1 style="
      margin:0;font-size:1.5rem;font-weight:600;letter-spacing:-0.03em;
      background:linear-gradient(135deg,#a78bfa,#22d3ee);
      -webkit-background-clip:text;-webkit-text-fill-color:transparent;">
      Smart FAQ Chatbot
    </h1>
    <p style="margin:4px 0 0;font-size:0.8rem;color:#475569;">
      Retrieval-Augmented Generation · Business Intelligence Midterm
    </p>
  </div>
  <div style="
    font-size:0.72rem;font-weight:500;color:#64748b;
    background:rgba(255,255,255,0.03);
    border:1px solid rgba(148,163,184,0.1);
    border-radius:20px;padding:5px 12px;">
    {status_dot}
  </div>
</div>
""", unsafe_allow_html=True)

SUGGESTIONS = [
    "What pricing plans are available?",
    "How secure is my data?",
    "What is Business Intelligence?",
    "Do you offer a free trial?",
    "How do I contact support?",
    "What data connectors are supported?",
]

# ── Chat messages ──────────────────────────────────────────────────────────────
if not st.session_state.messages:
    # Welcome state
    st.markdown("""
<div style="
  text-align:center;padding:60px 20px 40px;
  max-width:520px;margin:0 auto;">
  <div style="
    width:56px;height:56px;border-radius:16px;
    background:linear-gradient(135deg,#7c5cfc,#22d3ee);
    display:flex;align-items:center;justify-content:center;
    font-size:1.4rem;margin:0 auto 20px;">💬</div>
  <h3 style="color:#e2e8f0;font-size:1.15rem;font-weight:600;
letter-spacing:-0.02em;margin-bottom:8px;">
    Ask anything about the knowledge base
  </h3>
  <p style="color:#475569;font-size:0.85rem;line-height:1.7;">
    Upload FAQ documents or load the sample, then start chatting.<br>
    The AI searches your docs and generates accurate answers.
  </p>
</div>
""", unsafe_allow_html=True)

    if kb_ready:
        st.markdown("""<p style="text-align:center;color:#334155;
font-size:0.75rem;font-weight:600;letter-spacing:0.08em;
text-transform:uppercase;margin-bottom:12px;">Try asking</p>""",
                    unsafe_allow_html=True)
        cols = st.columns(3)
        for i, sug in enumerate(SUGGESTIONS):
            with cols[i % 3]:
                if st.button(sug, key=f"sug_{i}", use_container_width=True):
                    st.session_state.pending_question = sug
                    st.rerun()
else:
    for msg in st.session_state.messages:
        avatar = "🤖" if msg["role"] == "assistant" else "👤"
        with st.chat_message(msg["role"], avatar=avatar):
            st.markdown(msg["content"])
            sources = msg.get("sources", [])
            if sources:
                with st.expander(f"↗ {len(sources)} source{'s' if len(sources)>1 else ''} retrieved"):
                    for chunk, score in sources:
                        st.markdown(
                            f"**{chunk['source']}** &nbsp;·&nbsp; "
                            f"<span style='color:#7c5cfc;font-size:0.78rem'>{score:.0%} match</span>",
                            unsafe_allow_html=True,
                        )
                        st.caption(chunk["text"][:280] + ("…" if len(chunk["text"]) > 280 else ""))
                        st.divider()
            if msg.get("timestamp"):
                st.markdown(
                    f"<div style='color:#334155;font-size:0.68rem;margin-top:4px;'>"
                    f"{msg['timestamp']}</div>",
                    unsafe_allow_html=True,
                )

# ── Chat input ─────────────────────────────────────────────────────────────────
if not kb_ready:
    st.markdown("""
<div style="text-align:center;padding:20px;color:#475569;font-size:0.85rem;">
  Load the <strong style="color:#64748b">Sample FAQ</strong> or upload documents
  in the sidebar to start chatting.
</div>""", unsafe_allow_html=True)
else:
    user_input = st.chat_input("Ask a question…")

    if st.session_state.pending_question:
        user_input = st.session_state.pending_question
        st.session_state.pending_question = ""

    if user_input and user_input.strip():
        question = user_input.strip()

        st.session_state.messages.append({
            "role": "user",
            "content": question,
            "timestamp": _ts(),
        })

        history = [
            {"role": m["role"], "content": m["content"]}
            for m in st.session_state.messages[:-1]
        ]

        with st.spinner(""):
            try:
                answer, sources = rag_query(
                    question=question,
                    vector_store=kb,
                    conversation_history=history,
                    api_key=GROQ_API_KEY,
                )
            except Exception as e:
                answer = f"Something went wrong: {e}"
                sources = []

        st.session_state.messages.append({
            "role": "assistant",
            "content": answer,
            "timestamp": _ts(),
            "sources": sources,
        })

        st.session_state.conversation_log.append({
            "question": question,
            "timestamp": datetime.now().strftime("%d %b %H:%M"),
        })

        st.rerun()
