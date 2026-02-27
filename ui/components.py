"""
UI component functions — each renders one logical section.
Following SRP: separate functions for separate concerns.
Following DRY: reusable across pages if app grows.
"""

import os
import tempfile
from pathlib import Path
from typing import Tuple, Optional

import streamlit as st

from config import config
from services.rag_service import RAGService
from ui.helpers import (
    display_answer, display_retrieved_context,
    display_indexed_summary, cleanup_temp_file,
)


def render_sidebar(service: RAGService) -> str:
    """Render sidebar and return the currently selected model name."""
    with st.sidebar:
        st.header("⚙️ Configuration")

        selected_model = st.selectbox(
            "Gemini Model",
            config.model.available_models,
            index=0,
        )
        service.set_model(selected_model)

        st.divider()
        st.subheader("📊 Index Stats")
        st.write(f"Indexed chunks: **{service.chunk_count}**")
        supported = ", ".join(f".{e}" for e in service.supported_extensions)
        st.caption(f"Supported: {supported}")

        st.divider()
        if st.button("🗑️ Clear All Documents", use_container_width=True):
            service.reset()
            st.session_state.clear()
            st.rerun()

    return selected_model


def render_upload_section(service: RAGService) -> None:
    """Render the file upload panel. Handles multiple files."""
    st.header("📂 Upload Documents")

    ext_list = [f".{e}" for e in service.supported_extensions]
    uploaded_files = st.file_uploader(
        "Upload files (PDF, Word, PowerPoint, Excel, Images)",
        type=service.supported_extensions,
        accept_multiple_files=True,
        help=f"Supported formats: {', '.join(ext_list)}",
    )

    if not uploaded_files:
        return

    for uploaded_file in uploaded_files:
        # Avoid reprocessing the same file
        cache_key = f"processed_{uploaded_file.name}_{uploaded_file.size}"
        if st.session_state.get(cache_key):
            st.success(f"✅ Already indexed: {uploaded_file.name}")
            continue

        with st.spinner(f"⚙️ Processing `{uploaded_file.name}`…"):
            suffix = Path(uploaded_file.name).suffix
            with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
                tmp.write(uploaded_file.getbuffer())
                tmp_path = tmp.name

            try:
                n_chunks = service.ingest(tmp_path)
                st.success(f"✅ `{uploaded_file.name}` → {n_chunks} chunks indexed")
                st.session_state[cache_key] = True
            except Exception as e:
                st.error(f"❌ Failed to process `{uploaded_file.name}`: {e}")
                if config.debug:
                    st.exception(e)
            finally:
                cleanup_temp_file(tmp_path)

    if service.chunk_count > 0:
        with st.expander("📊 Index Summary"):
            display_indexed_summary(service.indexed_chunks)


def render_qa_section(service: RAGService) -> None:
    """Render the question-answering panel."""
    st.header("💬 Ask Questions")

    if service.chunk_count == 0:
        st.info("👈 Upload documents on the left to begin.")
        _render_tips()
        return

    _render_example_questions()

    user_query = st.text_area(
        "Your question:",
        height=100,
        value=st.session_state.get("query", ""),
        placeholder="e.g., What are the key findings in the report?",
    )

    if st.button("🔍 Get Answer", type="primary", use_container_width=True):
        _handle_query(service, user_query)


def _handle_query(service: RAGService, query: str) -> None:
    """Run the RAG pipeline and display results."""
    if not query.strip():
        st.warning("Please enter a question.")
        return

    with st.spinner("🤔 Analyzing documents with Gemini…"):
        response = service.answer(query)

    st.session_state["query"] = query
    display_answer(response)
    display_retrieved_context(response.results)


def _render_example_questions() -> None:
    """Show clickable example questions."""
    st.markdown("**Quick questions:**")
    examples = [
        "What is this document about?",
        "Summarize the key points.",
        "What data or numbers are mentioned?",
        "Describe any charts or images.",
    ]
    cols = st.columns(2)
    for i, q in enumerate(examples):
        if cols[i % 2].button(q, key=f"ex_{i}"):
            st.session_state["query"] = q
            st.rerun()


def _render_tips() -> None:
    st.markdown("### 💡 Tips")
    st.markdown(
        "- Upload **PDF, Word, PowerPoint, Excel**, or **image** files\n"
        "- Images inside PDFs are automatically extracted and analyzed\n"
        "- Ask specific questions for better results\n"
        "- You can upload **multiple files** at once"
    )