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

# ── Page configuration (must be first Streamlit call) ──────────────────────
st.set_page_config(
    page_title="Smart FAQ Chatbot",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded",
    menu_items={
        "Get Help": "mailto:support@techcorp.ai",
        "About": "Smart FAQ Chatbot powered by RAG — BI Midterm Project",
    },
)

# ── Imports after page config ───────────────────────────────────────────────
from utils.document_processor import process_document
from utils.vector_store import VectorStore
from utils.rag_pipeline import get_embeddings, query as rag_query

# ── Custom CSS ──────────────────────────────────────────────────────────────
st.markdown("""
<style>
  /* ---- Global ---- */
  [data-testid="stAppViewContainer"] { background: #0f1117; }
  [data-testid="stSidebar"] { background: #1a1d27; border-right: 1px solid #2d3145; }
  .stTextInput > div > div > input { background: #1e2130; border: 1px solid #3a3f5c; color: #e0e4f0; }

  /* ---- Header ---- */
  .app-header {
    background: linear-gradient(135deg, #1e2130 0%, #252a40 100%);
    border: 1px solid #3a3f5c;
    border-radius: 12px;
    padding: 20px 28px;
    margin-bottom: 24px;
    display: flex;
    align-items: center;
    gap: 14px;
  }
  .app-header h1 { margin: 0; font-size: 1.6rem; color: #e0e4f0; }
  .app-header p  { margin: 4px 0 0; font-size: 0.85rem; color: #8892b0; }

  /* ---- Status badge ---- */
  .status-badge {
    display: inline-flex;
    align-items: center;
    gap: 6px;
    padding: 4px 12px;
    border-radius: 20px;
    font-size: 0.78rem;
    font-weight: 600;
    margin-left: auto;
  }
  .status-ready   { background: rgba(0,230,118,.12); color: #00e676; border: 1px solid rgba(0,230,118,.3); }
  .status-empty   { background: rgba(255,183,77,.10); color: #ffb74d; border: 1px solid rgba(255,183,77,.25); }

  /* ---- Chat bubbles ---- */
  .chat-container { display: flex; flex-direction: column; gap: 16px; padding-bottom: 8px; }

  .msg-row { display: flex; align-items: flex-start; gap: 10px; }
  .msg-row.user    { flex-direction: row-reverse; }
  .msg-row.user .bubble    { background: #2e3f6e; border: 1px solid #3b54a0; border-radius: 18px 4px 18px 18px; }
  .msg-row.assistant .bubble { background: #1e2130; border: 1px solid #2d3145; border-radius: 4px 18px 18px 18px; }

  .avatar {
    width: 36px; height: 36px; border-radius: 50%;
    display: flex; align-items: center; justify-content: center;
    font-size: 1.05rem; flex-shrink: 0; margin-top: 2px;
  }
  .avatar.user      { background: #2e3f6e; border: 2px solid #3b54a0; }
  .avatar.assistant { background: #1e2130; border: 2px solid #3a3f5c; }

  .bubble { padding: 12px 16px; max-width: 75%; color: #e0e4f0; line-height: 1.6; }
  .bubble p { margin: 0 0 8px; }
  .bubble p:last-child { margin-bottom: 0; }

  .msg-meta { font-size: 0.72rem; color: #5a6080; margin-top: 4px; text-align: right; }
  .msg-row.assistant .msg-meta { text-align: left; }

  /* ---- Sources pill ---- */
  .source-pill {
    display: inline-block;
    background: rgba(99,120,222,.15);
    border: 1px solid rgba(99,120,222,.35);
    border-radius: 4px;
    padding: 2px 8px;
    font-size: 0.72rem;
    color: #8899dd;
    margin: 2px 3px 2px 0;
  }

  /* ---- Sidebar ---- */
  .sidebar-section { margin-bottom: 8px; }
  .sidebar-title {
    font-size: 0.72rem;
    font-weight: 700;
    letter-spacing: .08em;
    text-transform: uppercase;
    color: #5a6080;
    margin-bottom: 8px;
  }
  .kb-stat {
    background: #1e2130;
    border: 1px solid #2d3145;
    border-radius: 8px;
    padding: 10px 14px;
    margin-bottom: 8px;
  }
  .kb-stat .stat-num  { font-size: 1.5rem; font-weight: 700; color: #6378de; }
  .kb-stat .stat-label{ font-size: 0.78rem; color: #8892b0; }

  .history-item {
    background: #1e2130;
    border: 1px solid #2d3145;
    border-radius: 8px;
    padding: 8px 12px;
    margin-bottom: 6px;
    cursor: pointer;
    transition: border-color .2s;
  }
  .history-item:hover { border-color: #6378de; }
  .history-q  { font-size: 0.82rem; color: #c8cde0; white-space: nowrap; overflow: hidden; text-overflow: ellipsis; }
  .history-ts { font-size: 0.68rem; color: #5a6080; margin-top: 2px; }

  /* ── Welcome card ── */
  .welcome-card {
    background: #1e2130;
    border: 1px dashed #3a3f5c;
    border-radius: 12px;
    padding: 40px 32px;
    text-align: center;
    color: #8892b0;
  }
  .welcome-card .icon  { font-size: 3rem; margin-bottom: 12px; }
  .welcome-card h3     { color: #c8cde0; margin-bottom: 8px; }
  .welcome-card p      { font-size: 0.88rem; line-height: 1.6; }

  /* ── Suggestion chips ── */
  .chip-row { display: flex; flex-wrap: wrap; gap: 8px; justify-content: center; margin-top: 20px; }
  .chip {
    background: #1a1d27;
    border: 1px solid #3a3f5c;
    border-radius: 20px;
    padding: 6px 14px;
    font-size: 0.80rem;
    color: #8892b0;
    cursor: pointer;
  }

  /* Streamlit overrides */
  div[data-testid="stFileUploaderDropzone"] { background: #1e2130 !important; border-color: #3a3f5c !important; }
  .stButton > button { border-radius: 8px !important; }
  .stButton > button[kind="primary"] { background: #3b54a0 !important; border-color: #3b54a0 !important; }
</style>
""", unsafe_allow_html=True)


# ── Session state initialisation ────────────────────────────────────────────
def _init():
    defaults = {
        "messages": [],
        "vector_store": VectorStore(),
        "api_key": os.environ.get("OPENAI_API_KEY", ""),
        "indexed_docs": [],          # list of doc names
        "conversation_log": [],       # sidebar history entries
        "pending_question": "",
    }
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v

_init()


# ── Helper: format timestamp ─────────────────────────────────────────────────
def _ts() -> str:
    return datetime.now().strftime("%H:%M")


# ── Helper: render a single chat message ─────────────────────────────────────
def _render_message(msg: Dict):
    role      = msg["role"]
    content   = msg["content"]
    timestamp = msg.get("timestamp", "")
    sources   = msg.get("sources", [])

    avatar = "👤" if role == "user" else "🤖"

    st.markdown(f"""
<div class="msg-row {role}">
  <div class="avatar {role}">{avatar}</div>
  <div>
    <div class="bubble">{_md_to_html(content)}</div>
    {'<div class="msg-meta">' + timestamp + '</div>' if timestamp else ''}
  </div>
</div>
""", unsafe_allow_html=True)

    # Show source references below assistant messages
    if role == "assistant" and sources:
        with st.expander(f"📎 {len(sources)} source(s) used", expanded=False):
            for chunk, score in sources:
                col1, col2 = st.columns([3, 1])
                with col1:
                    st.markdown(f"**{chunk['source']}**")
                    st.caption(chunk["text"][:300] + ("…" if len(chunk["text"]) > 300 else ""))
                with col2:
                    st.metric("Relevance", f"{score:.0%}")
                st.divider()


def _md_to_html(text: str) -> str:
    """Minimal markdown → HTML for bubble rendering (bold, lists, newlines)."""
    import re
    # Bold
    text = re.sub(r"\*\*(.+?)\*\*", r"<strong>\1</strong>", text)
    # Bullet lists
    lines = text.split("\n")
    in_list = False
    out = []
    for line in lines:
        if line.startswith("- ") or line.startswith("• "):
            if not in_list:
                out.append("<ul>")
                in_list = True
            out.append(f"<li>{line[2:].strip()}</li>")
        else:
            if in_list:
                out.append("</ul>")
                in_list = False
            if line.strip():
                out.append(f"<p>{line}</p>")
    if in_list:
        out.append("</ul>")
    return "".join(out)


# ── Helper: ingest a single uploaded file ────────────────────────────────────
def _ingest_file(file) -> int:
    """Process and index one uploaded file. Returns chunk count."""
    file_bytes = file.read()
    chunks = process_document(file_bytes, file.name)

    from openai import OpenAI
    client = OpenAI(api_key=st.session_state.api_key)
    texts = [c["text"] for c in chunks]
    embeddings = get_embeddings(texts, client)
    st.session_state.vector_store.add(chunks, embeddings)
    return len(chunks)


# ── Sidebar ──────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown('<div class="sidebar-title">⚙️ Configuration</div>', unsafe_allow_html=True)

    api_key_input = st.text_input(
        "OpenAI API Key",
        type="password",
        value=st.session_state.api_key,
        placeholder="sk-...",
        help="Get your key at platform.openai.com",
    )
    if api_key_input:
        st.session_state.api_key = api_key_input

    api_ok = bool(st.session_state.api_key and st.session_state.api_key.startswith("sk-"))
    if api_ok:
        st.success("API key set ✓", icon="🔑")
    else:
        st.warning("Enter your OpenAI API key to start.", icon="⚠️")

    st.divider()

    # ── Knowledge Base ──────────────────────────────────────────────────────
    st.markdown('<div class="sidebar-title">📚 Knowledge Base</div>', unsafe_allow_html=True)

    kb: VectorStore = st.session_state.vector_store
    col_a, col_b = st.columns(2)
    with col_a:
        st.markdown(f"""
<div class="kb-stat">
  <div class="stat-num">{kb.num_chunks}</div>
  <div class="stat-label">Chunks indexed</div>
</div>""", unsafe_allow_html=True)
    with col_b:
        st.markdown(f"""
<div class="kb-stat">
  <div class="stat-num">{len(kb.sources)}</div>
  <div class="stat-label">Documents</div>
</div>""", unsafe_allow_html=True)

    uploaded_files = st.file_uploader(
        "Upload FAQ documents",
        type=["pdf", "txt", "docx"],
        accept_multiple_files=True,
        help="Supports PDF, TXT, and DOCX formats",
        label_visibility="collapsed",
    )

    # Load sample data shortcut
    sample_col, upload_col = st.columns(2)
    with sample_col:
        if st.button("Load Sample FAQ", use_container_width=True, help="Load the built-in TechCorp FAQ"):
            if not api_ok:
                st.error("Set API key first.")
            else:
                with st.spinner("Indexing sample FAQ…"):
                    try:
                        with open("data/sample_faq.txt", "rb") as f:
                            sample_bytes = f.read()

                        # Remove old sample if present
                        kb.remove_source("sample_faq.txt")

                        from utils.document_processor import process_document as _pd
                        chunks = _pd(sample_bytes, "sample_faq.txt")
                        from openai import OpenAI
                        client = OpenAI(api_key=st.session_state.api_key)
                        texts = [c["text"] for c in chunks]
                        embs  = get_embeddings(texts, client)
                        kb.add(chunks, embs)
                        if "sample_faq.txt" not in st.session_state.indexed_docs:
                            st.session_state.indexed_docs.append("sample_faq.txt")
                        st.success(f"Loaded {len(chunks)} chunks!")
                        st.rerun()
                    except Exception as e:
                        st.error(f"Error: {e}")

    with upload_col:
        process_btn = st.button(
            "Process Files",
            use_container_width=True,
            disabled=(not uploaded_files or not api_ok),
            type="primary",
        )

    if process_btn and uploaded_files:
        progress = st.progress(0, text="Starting…")
        total_chunks = 0
        for i, uf in enumerate(uploaded_files):
            progress.progress((i) / len(uploaded_files), text=f"Processing {uf.name}…")
            try:
                n = _ingest_file(uf)
                total_chunks += n
                if uf.name not in st.session_state.indexed_docs:
                    st.session_state.indexed_docs.append(uf.name)
            except Exception as e:
                st.error(f"❌ {uf.name}: {e}")
        progress.progress(1.0, text="Done!")
        time.sleep(0.8)
        progress.empty()
        st.success(f"✅ Indexed {total_chunks} chunks from {len(uploaded_files)} file(s)")
        st.rerun()

    # Indexed documents list
    if kb.sources:
        with st.expander("Indexed documents", expanded=False):
            for src in kb.sources:
                col1, col2 = st.columns([4, 1])
                with col1:
                    st.markdown(f"📄 `{src}`")
                with col2:
                    if st.button("✕", key=f"rm_{src}", help=f"Remove {src}"):
                        kb.remove_source(src)
                        if src in st.session_state.indexed_docs:
                            st.session_state.indexed_docs.remove(src)
                        st.rerun()

    st.divider()

    # ── Conversation History ────────────────────────────────────────────────
    st.markdown('<div class="sidebar-title">💬 Conversation History</div>', unsafe_allow_html=True)

    if st.session_state.conversation_log:
        for entry in reversed(st.session_state.conversation_log[-15:]):
            st.markdown(f"""
<div class="history-item">
  <div class="history-q">❓ {entry["question"][:55]}{"…" if len(entry["question"]) > 55 else ""}</div>
  <div class="history-ts">{entry["timestamp"]}</div>
</div>""", unsafe_allow_html=True)

        if st.button("Clear History", use_container_width=True):
            st.session_state.messages = []
            st.session_state.conversation_log = []
            st.rerun()
    else:
        st.caption("No conversations yet.")

    st.divider()
    st.caption("🤖 Smart FAQ Chatbot · RAG · BI Midterm")
    st.caption("Powered by OpenAI + Streamlit")


# ── Main content area ─────────────────────────────────────────────────────────
kb_ready = kb.num_chunks > 0

status_html = (
    '<span class="status-badge status-ready">● Knowledge Base Ready</span>'
    if kb_ready else
    '<span class="status-badge status-empty">● No Documents Loaded</span>'
)

st.markdown(f"""
<div class="app-header">
  <div style="font-size:2rem">🤖</div>
  <div>
    <h1>Smart FAQ Chatbot</h1>
    <p>Retrieval-Augmented Generation · Business Intelligence Midterm</p>
  </div>
  {status_html}
</div>
""", unsafe_allow_html=True)

# ── Suggested questions (shown when KB is loaded but no messages) ─────────────
SUGGESTIONS = [
    "What pricing plans are available?",
    "How secure is my data?",
    "What is Business Intelligence?",
    "Do you offer a free trial?",
    "How do I contact support?",
    "What data connectors are supported?",
]

# ── Chat messages ─────────────────────────────────────────────────────────────
chat_container = st.container()

with chat_container:
    if not st.session_state.messages:
        st.markdown("""
<div class="welcome-card">
  <div class="icon">💬</div>
  <h3>Ask anything about our FAQ</h3>
  <p>Upload your company's FAQ documents in the sidebar, then ask questions.<br>
  The AI will search the knowledge base and generate accurate answers.</p>
</div>
""", unsafe_allow_html=True)

        if kb_ready:
            st.markdown("**Try asking:**")
            cols = st.columns(3)
            for i, sug in enumerate(SUGGESTIONS):
                with cols[i % 3]:
                    if st.button(sug, key=f"sug_{i}", use_container_width=True):
                        st.session_state.pending_question = sug
                        st.rerun()
    else:
        st.markdown('<div class="chat-container">', unsafe_allow_html=True)
        for msg in st.session_state.messages:
            _render_message(msg)
        st.markdown("</div>", unsafe_allow_html=True)

# ── Chat input ────────────────────────────────────────────────────────────────
st.markdown("<br>", unsafe_allow_html=True)

if not api_ok:
    st.info("🔑 Please enter your OpenAI API key in the sidebar to start chatting.", icon="ℹ️")
elif not kb_ready:
    st.info("📚 Load the Sample FAQ or upload your own documents in the sidebar to begin.", icon="ℹ️")
else:
    user_input = st.chat_input(
        "Ask a question about the knowledge base…",
        key="chat_input",
    )

    # Handle suggestion button clicks
    if st.session_state.pending_question:
        user_input = st.session_state.pending_question
        st.session_state.pending_question = ""

    if user_input and user_input.strip():
        question = user_input.strip()

        # Add user message
        st.session_state.messages.append({
            "role": "user",
            "content": question,
            "timestamp": _ts(),
        })

        # Build conversation history for context (text only)
        history = [
            {"role": m["role"], "content": m["content"]}
            for m in st.session_state.messages[:-1]
        ]

        # Generate answer
        with st.spinner("Searching knowledge base and generating answer…"):
            try:
                answer, sources = rag_query(
                    question=question,
                    vector_store=kb,
                    conversation_history=history,
                    api_key=st.session_state.api_key,
                )
            except Exception as e:
                answer = f"⚠️ An error occurred: {e}"
                sources = []

        # Add assistant message
        st.session_state.messages.append({
            "role": "assistant",
            "content": answer,
            "timestamp": _ts(),
            "sources": sources,
        })

        # Update conversation log for sidebar history
        st.session_state.conversation_log.append({
            "question": question,
            "timestamp": datetime.now().strftime("%d %b %H:%M"),
        })

        st.rerun()
