"""
app.py — Multimodal RAG Q&A System entry point.

Run with: streamlit run app.py
"""

import streamlit as st

from config import config
from services.rag_service import RAGService
from ui.components import render_sidebar, render_upload_section, render_qa_section


def _init_session() -> None:
    """Initialize session-scoped RAGService (persists across Streamlit re-runs)."""
    if "rag_service" not in st.session_state:
        st.session_state["rag_service"] = RAGService()


def _check_api_key() -> bool:
    """Validate that the API key is configured."""
    if not config.api.google_api_key:
        st.error(
            "⚠️ **Google API key not found.**  \n"
            "Add `GOOGLE_API_KEY=your_key` to a `.env` file in the project root."
        )
        return False
    return True


def main() -> None:
    st.set_page_config(
        page_title=config.app_name,
        page_icon="🧠",
        layout="wide",
    )

    st.title("🧠 Multimodal RAG Q&A System")
    st.caption("Powered by Gemini · Supports PDF, Word, PowerPoint, Excel & Images")
    st.markdown("---")

    if not _check_api_key():
        return

    _init_session()
    service: RAGService = st.session_state["rag_service"]

    render_sidebar(service)

    col_upload, col_qa = st.columns([1, 1], gap="large")

    with col_upload:
        render_upload_section(service)

    with col_qa:
        render_qa_section(service)


if __name__ == "__main__":
    main()