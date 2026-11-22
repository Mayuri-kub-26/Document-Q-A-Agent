"""ChatGPT-style Streamlit frontend for the Document Q&A (RAG) API."""

import os
from uuid import uuid4
from typing import Any, Dict, List, Optional

import httpx
import streamlit as st

st.set_page_config(page_title="Document Q&A Agent", page_icon="💬", layout="wide")

DEFAULT_BACKEND = os.getenv("BACKEND_URL", "http://localhost:8000")

st.markdown(
    """
    <style>
        #MainMenu, footer, header {visibility: hidden;}
        .stApp { background: #ffffff; }
        [data-testid="stSidebar"] {
            background: #f7f7f8;
            border-right: 1px solid #e6e6e6;
        }
        [data-testid="stSidebar"] .stButton button {
            text-align: left;
            border-radius: 10px;
        }
        .block-container {
            max-width: 840px;
            padding-top: 2.4rem;
            padding-bottom: 9rem;
        }
        .stChatMessage {
            background: transparent;
            border: none;
            padding: 0;
            margin-bottom: 1.5rem;
        }
        .stChatMessage .stMarkdown {
            font-size: 15.5px;
            line-height: 1.65;
        }
        .welcome {
            text-align: center;
            margin-top: 7vh;
        }
        .welcome h1 {
            font-size: 2.4rem;
            font-weight: 600;
            letter-spacing: -0.02em;
            margin-bottom: 0.3rem;
        }
        .welcome p { color: #6b7280; font-size: 1.05rem; }
        .suggestion {
            border: 1px solid #e5e7eb;
            border-radius: 12px;
            padding: 0.9rem 1rem;
            background: #fff;
            text-align: left;
            min-height: 70px;
        }
        .suggestion:hover { background: #f7f7f8; border-color: #d1d5db; }
        .stButton button { border-radius: 10px; }
        [data-testid="stChatInput"] { background: #ffffff; }
    </style>
    """,
    unsafe_allow_html=True,
)


# --------------------------------------------------------------------------- #
# Session state
# --------------------------------------------------------------------------- #
if "conversations" not in st.session_state:
    _cid = uuid4().hex
    st.session_state.conversations = {_cid: {"title": "New chat", "messages": []}}
    st.session_state.active_conv = _cid

if "top_k" not in st.session_state:
    st.session_state.top_k = 4
if "show_sources" not in st.session_state:
    st.session_state.show_sources = True
if "use_web" not in st.session_state:
    st.session_state.use_web = True
if "backend_url" not in st.session_state:
    st.session_state.backend_url = DEFAULT_BACKEND


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


def show_sources_widget(sources: List[Dict[str, Any]]) -> None:
    if not sources:
        return
    with st.expander(f"📎 {len(sources)} source chunk(s)", expanded=False):
        for i, src in enumerate(sources, 1):
            meta = src.get("metadata") or {}
            label = meta.get("source") or src.get("id", f"chunk-{i}")
            st.markdown(f"**[{i}] {label}** · score `{src.get('score')}`")
            if meta.get("url"):
                st.markdown(f"🔗 {meta['url']}")
            st.caption(src.get("text", "")[:600])
            if i != len(sources):
                st.divider()


@st.dialog("📎 Upload documents")
def upload_dialog() -> None:
    st.caption("Add files to the knowledge base. The assistant can then answer from them.")
    uploaded = st.file_uploader(
        "Choose a document",
        type=["txt", "md", "pdf", "docx"],
        help="Supported: .txt, .md, .pdf, .docx (max 10MB)",
        key="dialog_uploader",
    )
    source_name = st.text_input("Source name (optional)", key="dialog_source")

    if st.button("⬆️ Upload & index", type="primary", disabled=uploaded is None, key="dialog_do_upload"):
        if uploaded is not None:
            with st.spinner("Uploading and indexing…"):
                try:
                    files = {
                        "file": (
                            uploaded.name,
                            uploaded.getvalue(),
                            uploaded.type or "application/octet-stream",
                        )
                    }
                    data = {"source": source_name.strip()} if source_name.strip() else {}
                    resp = httpx.post(api("/ingest/file"), files=files, data=data, timeout=120.0)
                    if resp.status_code == 200:
                        result = resp.json()
                        st.success(
                            f"Indexed **{result.get('chunks_added')}** chunk(s) from "
                            f"`{result.get('source')}`."
                        )
                    else:
                        st.error(f"Upload failed ({resp.status_code}): {resp.json().get('detail', resp.text)}")
                except Exception as exc:
                    st.error(f"Upload failed: {exc}")

    st.divider()
    st.subheader("Or paste raw text")
    raw_text = st.text_area("Text to index", height=140, key="dialog_raw_text")
    raw_source = st.text_input("Source name", value="pasted_text", key="dialog_raw_source")
    if st.button("📥 Index text", disabled=not raw_text.strip(), key="dialog_do_text"):
        with st.spinner("Indexing text…"):
            try:
                resp = httpx.post(
                    api("/ingest/text"),
                    json={"text": raw_text, "source": raw_source},
                    timeout=120.0,
                )
                if resp.status_code == 200:
                    st.success(f"Indexed **{resp.json().get('chunks_added')}** chunk(s).")
                else:
                    st.error(f"Indexing failed: {resp.json().get('detail', resp.text)}")
            except Exception as exc:
                st.error(f"Indexing failed: {exc}")


def start_new_chat() -> None:
    cid = uuid4().hex
    st.session_state.conversations[cid] = {"title": "New chat", "messages": []}
    st.session_state.active_conv = cid
    st.rerun()


# --------------------------------------------------------------------------- #
# Sidebar
# --------------------------------------------------------------------------- #
with st.sidebar:
    if st.button("➕ New chat", use_container_width=True, type="primary", key="new_chat_btn"):
        start_new_chat()

    st.caption("Conversations")
    recent_ids = list(st.session_state.conversations.keys())[-12:][::-1]
    for cid in recent_ids:
        conv_item = st.session_state.conversations[cid]
        is_active = cid == st.session_state.active_conv
        if st.button(
            conv_item["title"],
            key=f"conv_{cid}",
            use_container_width=True,
            type="primary" if is_active else "secondary",
        ):
            st.session_state.active_conv = cid
            st.rerun()

    st.divider()
    if st.button("📎 Upload documents", use_container_width=True, key="open_upload_dialog"):
        upload_dialog()

    with st.expander("⚙️ Settings"):
        st.text_input("Backend URL", key="backend_url")
        st.slider("Sources to retrieve", 1, 10, 4, key="top_k")
        st.toggle("Show source chunks", key="show_sources")
        st.toggle("Use live web data", key="use_web")

        st.divider()
        st.caption("Danger zone")
        confirm_reset = st.checkbox("I understand, reset the vector store", key="confirm_reset")
        if st.button("🗑️ Reset store", disabled=not confirm_reset, key="reset_store_btn"):
            try:
                resp = httpx.post(api("/reset"), timeout=60.0)
                if resp.status_code == 200:
                    st.success("Vector store reset.")
                    st.rerun()
                else:
                    st.error(f"Reset failed: {resp.text}")
            except Exception as exc:
                st.error(f"Reset failed: {exc}")

    health = get_health()
    st.divider()
    if health:
        st.success(f"● Connected · {health.get('documents')} doc(s)")
        st.caption(f"Model: `{health.get('provider')}`")
    else:
        st.error("● Backend offline — start uvicorn first")


# --------------------------------------------------------------------------- #
# Main chat area
# --------------------------------------------------------------------------- #
active = st.session_state.conversations[st.session_state.active_conv]
messages: List[Dict[str, Any]] = active["messages"]

if not messages:
    st.markdown(
        """
        <div class="welcome">
            <h1>What can I help with?</h1>
            <p>Ask questions grounded in your own uploaded documents.</p>
        </div>
        """,
        unsafe_allow_html=True,
    )
    c1, c2 = st.columns(2)
    with c1:
        if st.button("📄 Summarize my documents", key="sug_summarize", use_container_width=True):
            st.session_state.pending_prompt = "Summarize the uploaded documents."
            st.rerun()
        if st.button("🔍 What documents are available?", key="sug_docs", use_container_width=True):
            st.session_state.pending_prompt = "What documents are available in the knowledge base?"
            st.rerun()
    with c2:
        if st.button("💡 Explain a key concept", key="sug_explain", use_container_width=True):
            st.session_state.pending_prompt = "Explain the main concepts found in the documents."
            st.rerun()
        if st.button("📎 Upload a document", key="sug_upload", use_container_width=True):
            upload_dialog()

for msg in messages:
    avatar = "🧑" if msg["role"] == "user" else "🤖"
    with st.chat_message(msg["role"], avatar=avatar):
        st.markdown(msg["content"])
        if msg.get("sources") and st.session_state.show_sources:
            show_sources_widget(msg["sources"])

prompt = st.chat_input("Message Document Q&A…")
if "pending_prompt" in st.session_state:
    prompt = st.session_state.pop("pending_prompt")

if prompt:
    # Build history from the current conversation before adding the new turn.
    history = [
        {"role": m["role"], "content": m["content"]}
        for m in messages
        if m["role"] in ("user", "assistant")
    ]

    # Save user message
    active["messages"].append({"role": "user", "content": prompt})
    if active["title"] == "New chat":
        active["title"] = (prompt[:32] + "…") if len(prompt) > 32 else prompt

    with st.chat_message("user", avatar="🧑"):
        st.markdown(prompt)

    with st.chat_message("assistant", avatar="🤖"):
        with st.spinner("Thinking…"):
            try:
                resp = httpx.post(
                    api("/query"),
                    json={
                        "question": prompt,
                        "top_k": st.session_state.top_k,
                        "include_sources": st.session_state.show_sources,
                        "use_web": st.session_state.use_web,
                        "history": history,
                    },
                    timeout=120.0,
                )
                if resp.status_code == 200:
                    data = resp.json()
                    answer = data.get("answer", "")
                    sources = data.get("sources", [])
                    active["messages"].append(
                        {
                            "role": "assistant",
                            "content": answer,
                            "sources": sources,
                            "model": data.get("model"),
                        }
                    )
                    st.markdown(answer)
                    if st.session_state.show_sources:
                        show_sources_widget(sources)
                else:
                    detail = resp.json().get("detail", resp.text)
                    st.error(f"Backend error ({resp.status_code}): {detail}")
            except Exception as exc:
                st.error(f"Could not reach backend: {exc}")
