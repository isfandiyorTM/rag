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

st.set_page_config(
    page_title="FAQ Assistant",
    page_icon="💬",
    layout="wide",
    initial_sidebar_state="expanded",
)

from utils.document_processor import process_document
from utils.vector_store import VectorStore
from utils.rag_pipeline import get_embeddings, query as rag_query

# ── CSS ────────────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600&display=swap');

html, body, [class*="css"] {
  font-family: 'Inter', -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif !important;
  -webkit-font-smoothing: antialiased;
}

#MainMenu, footer, [data-testid="stToolbar"] { display: none !important; }
header[data-testid="stHeader"]               { display: none !important; }

/* ── Layout backgrounds ── */
.stApp,
[data-testid="stAppViewContainer"],
[data-testid="stMain"] {
  background: #f4f4f5 !important;
}
[data-testid="stSidebar"] {
  background: #ffffff !important;
  border-right: 1px solid #e4e4e7 !important;
  box-shadow: none !important;
}
[data-testid="stSidebar"] > div:first-child { padding: 24px 18px !important; }
[data-testid="stBottom"],
[data-testid="stBottom"] > div {
  background: #f4f4f5 !important;
  border-top: 1px solid #e4e4e7 !important;
  padding-top: 8px !important;
}

/* ── Scrollbar ── */
::-webkit-scrollbar              { width: 4px; }
::-webkit-scrollbar-track        { background: transparent; }
::-webkit-scrollbar-thumb        { background: #d4d4d8; border-radius: 4px; }
::-webkit-scrollbar-thumb:hover  { background: #a1a1aa; }

/* ── Divider ── */
hr {
  border: none !important;
  border-top: 1px solid #f4f4f5 !important;
  margin: 16px 0 !important;
}

/* ── Text ── */
p, li, span { color: #3f3f46; }
h1, h2, h3  { color: #18181b; letter-spacing: -0.02em; }
.stCaption  { color: #a1a1aa !important; font-size: 0.72rem !important; }

/* ── Buttons ── */
.stButton > button {
  font-family: 'Inter', sans-serif !important;
  font-size: 0.8rem !important;
  font-weight: 500 !important;
  height: 34px !important;
  border-radius: 7px !important;
  border: 1px solid #e4e4e7 !important;
  background: #ffffff !important;
  color: #52525b !important;
  box-shadow: 0 1px 2px rgba(0,0,0,0.04) !important;
  transition: background 0.1s, box-shadow 0.1s !important;
  letter-spacing: 0 !important;
}
.stButton > button:hover {
  background: #fafafa !important;
  border-color: #d4d4d8 !important;
  color: #18181b !important;
  box-shadow: 0 1px 3px rgba(0,0,0,0.07) !important;
}
.stButton > button[kind="primary"] {
  background: #4f46e5 !important;
  border: none !important;
  color: #ffffff !important;
  box-shadow: 0 1px 3px rgba(79,70,229,0.25) !important;
}
.stButton > button[kind="primary"]:hover {
  background: #4338ca !important;
  box-shadow: 0 2px 8px rgba(79,70,229,0.3) !important;
}

/* ── Text / chat inputs ── */
.stTextInput > div > div > input {
  background: #fafafa !important;
  border: 1px solid #e4e4e7 !important;
  border-radius: 8px !important;
  color: #18181b !important;
  font-family: 'Inter', sans-serif !important;
  box-shadow: none !important;
  font-size: 0.875rem !important;
}
.stTextInput > div > div > input::placeholder { color: #a1a1aa !important; }
.stTextInput > div > div > input:focus {
  border-color: #4f46e5 !important;
  background: #ffffff !important;
  box-shadow: 0 0 0 3px rgba(79,70,229,0.1) !important;
}
.stTextInput label { color: #71717a !important; font-size: 0.74rem !important; font-weight: 500 !important; }

[data-testid="stChatInput"] {
  background: #ffffff !important;
  border: 1px solid #e4e4e7 !important;
  border-radius: 10px !important;
  box-shadow: 0 1px 4px rgba(0,0,0,0.06) !important;
}
[data-testid="stChatInput"]:focus-within {
  border-color: #4f46e5 !important;
  box-shadow: 0 0 0 3px rgba(79,70,229,0.08), 0 1px 4px rgba(0,0,0,0.06) !important;
}
[data-testid="stChatInput"] textarea {
  color: #18181b !important;
  background: transparent !important;
  font-family: 'Inter', sans-serif !important;
  font-size: 0.875rem !important;
}
[data-testid="stChatInput"] textarea::placeholder { color: #a1a1aa !important; }
[data-testid="stChatInputSubmitButton"] svg       { fill: #4f46e5 !important; }

/* ── Chat messages ── */
[data-testid="stChatMessage"] {
  padding: 2px 0 !important;
  gap: 10px !important;
  background: transparent !important;
  border: none !important;
  max-width: 860px !important;
  margin: 0 auto !important;
}
[data-testid="stChatMessageContent"] {
  background: #ffffff !important;
  border: 1px solid #e4e4e7 !important;
  border-radius: 3px 12px 12px 12px !important;
  padding: 11px 15px !important;
  color: #18181b !important;
  font-size: 0.875rem !important;
  line-height: 1.68 !important;
  box-shadow: 0 1px 2px rgba(0,0,0,0.05) !important;
  max-width: 76% !important;
}

/* user bubble */
[data-testid="stChatMessage"]:has([data-testid="stChatMessageAvatarUser"]) {
  flex-direction: row-reverse !important;
}
[data-testid="stChatMessage"]:has([data-testid="stChatMessageAvatarUser"])
  [data-testid="stChatMessageContent"] {
  background: #18181b !important;
  border: none !important;
  border-radius: 12px 3px 12px 12px !important;
  color: #fafafa !important;
  box-shadow: 0 1px 3px rgba(0,0,0,0.15) !important;
}
[data-testid="stChatMessage"]:has([data-testid="stChatMessageAvatarUser"])
  [data-testid="stChatMessageContent"] p,
[data-testid="stChatMessage"]:has([data-testid="stChatMessageAvatarUser"])
  [data-testid="stChatMessageContent"] li,
[data-testid="stChatMessage"]:has([data-testid="stChatMessageAvatarUser"])
  [data-testid="stChatMessageContent"] strong {
  color: #fafafa !important;
}

/* avatars */
[data-testid="stChatMessageAvatarUser"],
[data-testid="stChatMessageAvatarAssistant"] {
  border-radius: 7px !important;
  width: 30px !important;
  height: 30px !important;
  flex-shrink: 0 !important;
}
[data-testid="stChatMessageAvatarAssistant"] {
  background: #4f46e5 !important;
  border: none !important;
}
[data-testid="stChatMessageAvatarUser"] {
  background: #f4f4f5 !important;
  border: 1px solid #e4e4e7 !important;
}

/* markdown inside bubbles */
[data-testid="stChatMessageContent"] p                { margin: 0 0 5px !important; }
[data-testid="stChatMessageContent"] p:last-child      { margin-bottom: 0 !important; }
[data-testid="stChatMessageContent"] strong            { font-weight: 600 !important; }
[data-testid="stChatMessageContent"] ul,
[data-testid="stChatMessageContent"] ol               { padding-left: 16px !important; margin: 4px 0 !important; }
[data-testid="stChatMessageContent"] li               { margin: 2px 0 !important; }
[data-testid="stChatMessageContent"] code {
  background: #f4f4f5 !important;
  color: #4f46e5 !important;
  border-radius: 4px !important;
  padding: 1px 5px !important;
  font-size: 0.82rem !important;
}

/* ── File uploader ── */
[data-testid="stFileUploaderDropzone"] {
  background: #fafafa !important;
  border: 1.5px dashed #d4d4d8 !important;
  border-radius: 9px !important;
  transition: all 0.15s !important;
}
[data-testid="stFileUploaderDropzone"]:hover {
  border-color: #4f46e5 !important;
  background: #eef2ff !important;
}
[data-testid="stFileUploaderDropzoneInstructions"] { color: #a1a1aa !important; font-size: 0.78rem !important; }

/* ── Progress bar ── */
[data-testid="stProgressBar"] > div > div {
  background: #4f46e5 !important;
  border-radius: 3px !important;
}
[data-testid="stProgressBar"] { background: #e4e4e7 !important; border-radius: 3px !important; }

/* ── Expander ── */
[data-testid="stExpander"] {
  background: #fafafa !important;
  border: 1px solid #e4e4e7 !important;
  border-radius: 8px !important;
}
[data-testid="stExpander"] summary          { color: #71717a !important; font-size: 0.78rem !important; font-weight: 500 !important; }
[data-testid="stExpander"] summary:hover    { color: #3f3f46 !important; }

/* ── Alerts ── */
[data-testid="stAlert"] { border-radius: 8px !important; font-size: 0.875rem !important; }

/* ── Spinner ── */
[data-testid="stSpinner"] > div { border-top-color: #4f46e5 !important; }
</style>
""", unsafe_allow_html=True)


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

def _ts() -> str:
    return datetime.now().strftime("%H:%M")

def _ingest_file(file) -> int:
    file_bytes = file.read()
    chunks = process_document(file_bytes, file.name)
    embeddings = get_embeddings([c["text"] for c in chunks])
    st.session_state.vector_store.add(chunks, embeddings)
    return len(chunks)


# ── Sidebar ────────────────────────────────────────────────────────────────────
with st.sidebar:
    kb: VectorStore = st.session_state.vector_store

    # Brand
    st.markdown("""
<div style="display:flex;align-items:center;gap:9px;margin-bottom:22px;">
  <div style="width:32px;height:32px;background:#4f46e5;border-radius:8px;
    display:flex;align-items:center;justify-content:center;font-size:0.9rem;flex-shrink:0;">
    💬
  </div>
  <div>
    <div style="font-size:0.9rem;font-weight:600;color:#18181b;letter-spacing:-0.01em;">
      FAQ Assistant
    </div>
    <div style="font-size:0.68rem;color:#a1a1aa;margin-top:1px;">
      Powered by Groq
    </div>
  </div>
</div>
""", unsafe_allow_html=True)

    # Section label
    st.markdown("""<div style="font-size:0.68rem;font-weight:600;letter-spacing:0.09em;
color:#a1a1aa;text-transform:uppercase;margin-bottom:10px;">Knowledge Base</div>""",
                unsafe_allow_html=True)

    # Stats
    col_a, col_b = st.columns(2)
    with col_a:
        st.markdown(f"""
<div style="background:#f4f4f5;border-radius:8px;padding:9px 12px;">
  <div style="font-size:1.15rem;font-weight:600;color:#18181b;letter-spacing:-0.02em;">
    {kb.num_chunks}
  </div>
  <div style="font-size:0.68rem;color:#a1a1aa;margin-top:1px;">chunks</div>
</div>""", unsafe_allow_html=True)
    with col_b:
        st.markdown(f"""
<div style="background:#f4f4f5;border-radius:8px;padding:9px 12px;">
  <div style="font-size:1.15rem;font-weight:600;color:#18181b;letter-spacing:-0.02em;">
    {len(kb.sources)}
  </div>
  <div style="font-size:0.68rem;color:#a1a1aa;margin-top:1px;">documents</div>
</div>""", unsafe_allow_html=True)

    st.markdown("<div style='height:12px'></div>", unsafe_allow_html=True)

    uploaded_files = st.file_uploader(
        "Upload files",
        type=["pdf", "txt", "docx"],
        accept_multiple_files=True,
        label_visibility="collapsed",
    )

    c1, c2 = st.columns(2)
    with c1:
        load_sample = st.button("Load sample", use_container_width=True)
    with c2:
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
                kb.add(chunks, get_embeddings([c["text"] for c in chunks]))
                if "sample_faq.txt" not in st.session_state.indexed_docs:
                    st.session_state.indexed_docs.append("sample_faq.txt")
                st.toast(f"{len(chunks)} chunks indexed", icon="✅")
                st.rerun()
            except Exception as e:
                st.error(str(e))

    if process_btn and uploaded_files:
        bar = st.progress(0, text="")
        total = 0
        for i, uf in enumerate(uploaded_files):
            bar.progress(i / len(uploaded_files), text=uf.name)
            try:
                total += _ingest_file(uf)
                if uf.name not in st.session_state.indexed_docs:
                    st.session_state.indexed_docs.append(uf.name)
            except Exception as e:
                st.error(f"{uf.name}: {e}")
        bar.progress(1.0, text="Done")
        time.sleep(0.5)
        bar.empty()
        st.toast(f"{total} chunks from {len(uploaded_files)} file(s)", icon="✅")
        st.rerun()

    # Indexed docs
    if kb.sources:
        st.markdown("<div style='height:10px'></div>", unsafe_allow_html=True)
        for src in kb.sources:
            c1, c2 = st.columns([5, 1])
            with c1:
                label = src if len(src) <= 26 else src[:23] + "…"
                st.markdown(f"""<div style="font-size:0.75rem;color:#71717a;
padding:3px 0;display:flex;align-items:center;gap:5px;">
  <span style="color:#d4d4d8;">📄</span> {label}
</div>""", unsafe_allow_html=True)
            with c2:
                if st.button("✕", key=f"rm_{src}"):
                    kb.remove_source(src)
                    if src in st.session_state.indexed_docs:
                        st.session_state.indexed_docs.remove(src)
                    st.rerun()

    st.divider()

    # History
    st.markdown("""<div style="font-size:0.68rem;font-weight:600;letter-spacing:0.09em;
color:#a1a1aa;text-transform:uppercase;margin-bottom:10px;">History</div>""",
                unsafe_allow_html=True)

    log = st.session_state.conversation_log
    if log:
        for entry in reversed(log[-10:]):
            st.markdown(f"""
<div style="padding:7px 9px;border-radius:7px;background:#fafafa;
border:1px solid #f4f4f5;margin-bottom:5px;">
  <div style="font-size:0.78rem;color:#52525b;white-space:nowrap;
overflow:hidden;text-overflow:ellipsis;">
    {entry['question'][:46]}{'…' if len(entry['question']) > 46 else ''}
  </div>
  <div style="font-size:0.65rem;color:#a1a1aa;margin-top:2px;">{entry['timestamp']}</div>
</div>""", unsafe_allow_html=True)

        if st.button("Clear conversation", use_container_width=True):
            st.session_state.messages = []
            st.session_state.conversation_log = []
            st.rerun()
    else:
        st.markdown("""<div style="font-size:0.78rem;color:#a1a1aa;padding:4px 0;">
No conversations yet.</div>""", unsafe_allow_html=True)


# ── Guard ──────────────────────────────────────────────────────────────────────
if not GROQ_API_KEY:
    st.error(
        "**`GROQ_API_KEY` is not configured.** "
        "Add it under App Settings → Secrets on Streamlit Cloud, "
        "or set the `GROQ_API_KEY` environment variable locally.",
        icon="🔑",
    )
    st.stop()


# ── Main ───────────────────────────────────────────────────────────────────────
kb_ready = kb.num_chunks > 0

# Header
status_color = "#10b981" if kb_ready else "#d4d4d8"
status_label = "Ready" if kb_ready else "No documents"
st.markdown(f"""
<div style="display:flex;align-items:center;justify-content:space-between;
max-width:860px;margin:0 auto 28px;padding:0 2px;">
  <div>
    <h1 style="font-size:1.15rem;font-weight:600;color:#18181b;
margin:0;letter-spacing:-0.02em;">
      FAQ Assistant
    </h1>
    <p style="font-size:0.78rem;color:#a1a1aa;margin:3px 0 0;">
      Ask anything — I'll search the knowledge base and answer clearly.
    </p>
  </div>
  <div style="display:flex;align-items:center;gap:6px;
background:#ffffff;border:1px solid #e4e4e7;border-radius:20px;
padding:5px 12px;box-shadow:0 1px 2px rgba(0,0,0,0.04);">
    <div style="width:7px;height:7px;border-radius:50%;background:{status_color};"></div>
    <span style="font-size:0.72rem;font-weight:500;color:#71717a;">{status_label}</span>
  </div>
</div>
""", unsafe_allow_html=True)


SUGGESTIONS = [
    "What pricing plans are available?",
    "How secure is my data?",
    "What is Business Intelligence?",
    "Do you offer a free trial?",
    "How do I contact support?",
    "What data connectors do you support?",
]

# ── Messages ───────────────────────────────────────────────────────────────────
if not st.session_state.messages:
    st.markdown("""
<div style="text-align:center;padding:64px 20px 40px;max-width:460px;margin:0 auto;">
  <div style="width:44px;height:44px;background:#ffffff;border:1px solid #e4e4e7;
border-radius:11px;display:flex;align-items:center;justify-content:center;
font-size:1.1rem;margin:0 auto 16px;box-shadow:0 1px 3px rgba(0,0,0,0.06);">
    💬
  </div>
  <h2 style="font-size:1rem;font-weight:600;color:#18181b;margin:0 0 6px;
letter-spacing:-0.01em;">
    Start a conversation
  </h2>
  <p style="font-size:0.82rem;color:#71717a;line-height:1.65;margin:0;">
    Upload your FAQ documents in the sidebar, then ask questions.<br>
    I'll find the most relevant answers from your knowledge base.
  </p>
</div>
""", unsafe_allow_html=True)

    if kb_ready:
        st.markdown("""<p style="text-align:center;font-size:0.7rem;font-weight:600;
letter-spacing:0.08em;text-transform:uppercase;color:#a1a1aa;margin-bottom:10px;">
Suggested questions</p>""", unsafe_allow_html=True)
        cols = st.columns(3)
        for i, sug in enumerate(SUGGESTIONS):
            with cols[i % 3]:
                if st.button(sug, key=f"sug_{i}", use_container_width=True):
                    st.session_state.pending_question = sug
                    st.rerun()

else:
    # Wrap messages in a max-width container
    st.markdown('<div style="max-width:860px;margin:0 auto;">', unsafe_allow_html=True)
    for msg in st.session_state.messages:
        avatar = "🤖" if msg["role"] == "assistant" else "👤"
        with st.chat_message(msg["role"], avatar=avatar):
            st.markdown(msg["content"])
            sources = msg.get("sources", [])
            if sources:
                with st.expander(f"↗ {len(sources)} source{'s' if len(sources) > 1 else ''} used"):
                    for chunk, score in sources:
                        st.markdown(
                            f"**{chunk['source']}** &nbsp;"
                            f"<span style='color:#a1a1aa;font-size:0.75rem;'>"
                            f"{score:.0%} match</span>",
                            unsafe_allow_html=True,
                        )
                        st.caption(chunk["text"][:260] + ("…" if len(chunk["text"]) > 260 else ""))
                        if chunk != sources[-1][0]:
                            st.divider()
            if msg.get("timestamp"):
                st.markdown(
                    f"<div style='font-size:0.67rem;color:#a1a1aa;margin-top:5px;'>"
                    f"{msg['timestamp']}</div>",
                    unsafe_allow_html=True,
                )
    st.markdown("</div>", unsafe_allow_html=True)


# ── Input ──────────────────────────────────────────────────────────────────────
if not kb_ready:
    st.markdown("""
<div style="text-align:center;padding:16px;max-width:860px;margin:0 auto;">
  <span style="font-size:0.82rem;color:#a1a1aa;">
    Load the <strong style="color:#71717a;font-weight:500;">sample FAQ</strong>
    or upload documents in the sidebar to start chatting.
  </span>
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
