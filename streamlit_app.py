"""Streamlit frontend for the Document Q&A (RAG) API."""

import os
from typing import Any, Dict, List, Optional

import httpx
import streamlit as st

st.set_page_config(
    page_title="Document Q&A Agent",
    page_icon="📚",
    layout="wide",
    initial_sidebar_state="expanded",
)

DEFAULT_BACKEND = os.getenv("BACKEND_URL", "http://localhost:8000")

st.markdown(
    """
    <style>
        .block-container { padding-top: 2.2rem; padding-bottom: 3rem; }
        .stChatMessage {
            background: #f8fafc;
            border: 1px solid #e2e8f0;
            border-radius: 14px;
            padding: 0.8rem 1rem;
            margin-bottom: 0.6rem;
        }
        .stButton > button, .stDownloadButton > button { border-radius: 10px; }
        .stTabs [data-baseweb="tab"] { font-size: 16px; font-weight: 600; }
        .sidebar .sidebar-content { background: #f1f5f9; }
        h1 { margin-bottom: 0.2rem; }
        .subtitle { color: #64748b; margin-bottom: 1.5rem; }
    </style>
    """,
    unsafe_allow_html=True,
)


def backend_url() -> str:
    return str(st.session_state.get("backend_url", DEFAULT_BACKEND)).rstrip("/")


def api(path: str) -> str:
    return f"{backend_url()}{path}"


def get_health() -> Optional[Dict[str, Any]]:
    try:
        resp = httpx.get(api("/health"), timeout=5.0)
        if resp.status_code == 200:
            return resp.json()
    except Exception:
        return None
    return None


def show_sources(sources: List[Dict[str, Any]]) -> None:
    if not sources:
        return
    with st.expander(f"📎 {len(sources)} source chunk(s)", expanded=False):
        for i, src in enumerate(sources, 1):
            meta = src.get("metadata") or {}
            label = meta.get("source") or src.get("id", f"chunk-{i}")
            st.markdown(f"**[{i}] {label}** · score `{src.get('score')}`")
            st.caption(src.get("text", "")[:600])
            if i != len(sources):
                st.divider()


with st.sidebar:
    st.title("⚙️ Settings")
    st.text_input(
        "Backend URL",
        value=str(st.session_state.get("backend_url", DEFAULT_BACKEND)),
        key="backend_url",
        help="FastAPI server address. Use http://rag-api:8000 inside Docker.",
    )

    health = get_health()
    if health:
        st.success("● Backend connected")
        st.caption(f"Model: `{health.get('provider')}`")
        st.caption(f"Documents indexed: **{health.get('documents')}**")
    else:
        st.error("● Backend offline — start uvicorn first")

    st.divider()
    st.subheader("Query options")
    top_k = st.slider("Sources to retrieve", min_value=1, max_value=10, value=4)
    include_sources = st.toggle("Show source chunks", value=True)

    st.divider()
    if st.button("🧹 Clear chat", use_container_width=True):
        st.session_state.messages = []
        st.rerun()

st.markdown("# 📚 Document Q&A Agent")
st.markdown(
    '<div class="subtitle">Upload your documents, then ask questions grounded only in their content.</div>',
    unsafe_allow_html=True,
)

chat_tab, upload_tab, store_tab = st.tabs(["💬 Chat", "📄 Upload documents", "🗄️ Vector store"])


# --------------------------------------------------------------------------- #
# Chat
# --------------------------------------------------------------------------- #
with chat_tab:
    if "messages" not in st.session_state:
        st.session_state.messages = []

    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])
            if msg.get("sources") and include_sources:
                show_sources(msg["sources"])

    prompt = st.chat_input("Ask a question about your documents…")
    if prompt:
        with st.chat_message("user"):
            st.markdown(prompt)
        st.session_state.messages.append({"role": "user", "content": prompt})

        with st.chat_message("assistant"):
            with st.spinner("Searching documents and generating answer…"):
                try:
                    resp = httpx.post(
                        api("/query"),
                        json={
                            "question": prompt,
                            "top_k": top_k,
                            "include_sources": include_sources,
                        },
                        timeout=120.0,
                    )
                    if resp.status_code == 200:
                        data = resp.json()
                        answer = data.get("answer", "")
                        sources = data.get("sources", [])
                        st.session_state.messages.append(
                            {
                                "role": "assistant",
                                "content": answer,
                                "sources": sources,
                                "model": data.get("model"),
                            }
                        )
                        st.markdown(answer)
                        if include_sources:
                            show_sources(sources)
                    else:
                        detail = resp.json().get("detail", resp.text)
                        st.error(f"Backend error ({resp.status_code}): {detail}")
                except Exception as exc:
                    st.error(f"Could not reach backend: {exc}")


# --------------------------------------------------------------------------- #
# Upload
# --------------------------------------------------------------------------- #
with upload_tab:
    st.subheader("Upload a file")
    uploaded = st.file_uploader(
        "Choose a document",
        type=["txt", "md", "pdf", "docx"],
        help="Supported: .txt, .md, .pdf, .docx (max 10MB)",
    )
    source_name = st.text_input("Source name (optional)", placeholder="e.g. project_notes")

    if st.button("⬆️ Upload & index", type="primary", disabled=uploaded is None):
        if uploaded is not None:
            with st.spinner("Uploading and indexing…"):
                try:
                    files = {
                        "file": (uploaded.name, uploaded.getvalue(), uploaded.type or "application/octet-stream")
                    }
                    data = {"source": source_name.strip()} if source_name.strip() else {}
                    resp = httpx.post(api("/ingest/file"), files=files, data=data, timeout=120.0)
                    if resp.status_code == 200:
                        result = resp.json()
                        st.success(
                            f"Indexed **{result.get('chunks_added')}** chunk(s) from "
                            f"`{result.get('source')}`. You can now ask questions in the Chat tab."
                        )
                    else:
                        detail = resp.json().get("detail", resp.text)
                        st.error(f"Upload failed ({resp.status_code}): {detail}")
                except Exception as exc:
                    st.error(f"Upload failed: {exc}")

    st.divider()
    st.subheader("Or paste raw text")
    raw_text = st.text_area("Text to index", height=180, placeholder="Paste notes, article text, etc.")
    raw_source = st.text_input("Source name", value="pasted_text")

    if st.button("📥 Index text", disabled=not raw_text.strip()):
        with st.spinner("Indexing text…"):
            try:
                resp = httpx.post(
                    api("/ingest/text"),
                    json={"text": raw_text, "source": raw_source},
                    timeout=120.0,
                )
                if resp.status_code == 200:
                    result = resp.json()
                    st.success(f"Indexed **{result.get('chunks_added')}** chunk(s).")
                else:
                    detail = resp.json().get("detail", resp.text)
                    st.error(f"Indexing failed ({resp.status_code}): {detail}")
            except Exception as exc:
                st.error(f"Indexing failed: {exc}")


# --------------------------------------------------------------------------- #
# Vector store
# --------------------------------------------------------------------------- #
with store_tab:
    st.subheader("Current store")
    if health:
        st.json(health)
    else:
        st.warning("Health information unavailable.")

    st.divider()
    st.subheader("Danger zone")
    st.warning("This deletes all indexed documents. You will need to re-upload them.")
    confirm = st.checkbox("I understand, reset the vector store")
    if st.button("🗑️ Reset store", type="secondary", disabled=not confirm):
        try:
            resp = httpx.post(api("/reset"), timeout=60.0)
            if resp.status_code == 200:
                st.success("Vector store reset.")
                st.rerun()
            else:
                st.error(f"Reset failed: {resp.text}")
        except Exception as exc:
            st.error(f"Reset failed: {exc}")
