import tempfile
from pathlib import Path
from typing import Tuple

import streamlit as st

from config import config
from services.rag_service import RAGService
from ui.helpers import (
    display_answer,
    display_retrieved_context,
    display_indexed_summary,
    cleanup_temp_file,
)

def render_sidebar(service: RAGService) -> str:
    """Render sidebar with config, quota tracker, cache stats, and index info."""
    with st.sidebar:
        st.header("⚙️ Configuration")

        selected_model = st.selectbox(
            "Gemini Model",
            config.model.available_models,
            index=0,
            help="gemini-2.0-flash has 1500 free requests/day - recommended",
        )
        service.set_model(selected_model)

        st.divider()
        st.subheader("📊 API Quota (Free Tier)")
        stats = service.rate_limit_stats
        used  = stats["daily_used"]
        limit = stats["daily_limit"]
        pct   = used / limit if limit else 0

        st.progress(min(pct, 1.0), text=f"{used} / {limit} requests used today")

        col1, col2 = st.columns(2)
        col1.metric("Remaining",  stats["daily_remaining"])
        col2.metric("Resets in",  f"{stats['resets_in_hours']}h")
        st.caption(
            f"This minute: {stats['rpm_used']} / {stats['rpm_limit']} RPM"
        )

        st.divider()
        st.subheader("⚡ Answer Cache")
        st.metric("Cached answers", service.cache_size)
        st.caption(
            "Repeated questions are answered instantly from cache "
            "no API call used."
        )

        st.divider()
        st.subheader("🖼️ Image Processing")

        if service.ocr_available:
            st.success("✅ Tesseract OCR active, text images processed free")
        else:
            st.warning(
                "⚠️ Tesseract not found. Images will use Gemini Vision.\n\n"
                "Install: https://github.com/UB-Mannheim/tesseract/wiki"
            )

        caption_enabled = st.toggle(
            "Enable image captioning",
            value=True,
            help=(
                "Disable to save API quota entirely. "
                "Images won't be described but text still works."
            ),
        )
        service.set_skip_captioning(not caption_enabled)
        st.caption(f"Caption cache: {service.caption_cache_size} images stored on disk")

        st.divider()
        st.subheader("🗂️ Index")
        st.write(f"Indexed chunks: **{service.chunk_count}**")
        supported = ", ".join(f".{e}" for e in service.supported_extensions)
        st.caption(f"Supported formats: {supported}")

        st.divider()
        if st.button("🗑️ Clear All Documents", use_container_width=True):
            service.reset()
            st.session_state.clear()
            st.rerun()

    return selected_model


def render_upload_section(service: RAGService) -> None:
    """Render the multi-file upload panel and trigger ingestion."""
    st.header("📂 Upload Documents")

    uploaded_files = st.file_uploader(
        "Upload files (PDF, Word, PowerPoint, Excel, Images)",
        type=service.supported_extensions,
        accept_multiple_files=True,
        help=", ".join(f".{e}" for e in service.supported_extensions),
    )

    if not uploaded_files:
        return

    for uploaded_file in uploaded_files:
        _process_single_upload(uploaded_file, service)

    if service.chunk_count > 0:
        with st.expander("📊 Index Summary"):
            display_indexed_summary(service.indexed_chunks)


def _process_single_upload(uploaded_file, service: RAGService) -> None:
    """Handle ingestion for one uploaded file."""
    # Avoid reprocessing the same file within the session
    cache_key = f"processed_{uploaded_file.name}_{uploaded_file.size}"
    if st.session_state.get(cache_key):
        st.success(f"✅ Already indexed: **{uploaded_file.name}**")
        return

    suffix = Path(uploaded_file.name).suffix

    with st.spinner(f"⚙️ Processing `{uploaded_file.name}`…"):
        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
            tmp.write(uploaded_file.getbuffer())
            tmp_path = tmp.name

        try:
            n_chunks, img_stats = service.ingest(tmp_path)
            st.success(
                f"✅ **{uploaded_file.name}** — {n_chunks} chunks indexed"
            )
            st.session_state[cache_key] = True
            _display_image_stats(img_stats)

        except ValueError as e:
            # Unsupported file type
            st.error(f"❌ {e}")
        except Exception as e:
            st.error(f"❌ Failed to process `{uploaded_file.name}`: {e}")
            if config.debug:
                st.exception(e)
        finally:
            cleanup_temp_file(tmp_path)


def _display_image_stats(img_stats: dict) -> None:
    """Show image processing breakdown after a file is ingested."""
    if not img_stats or img_stats.get("total_images", 0) == 0:
        return

    total    = img_stats["total_images"]
    cached   = img_stats.get("from_cache", 0)
    from_ocr = img_stats.get("from_ocr",   0)
    vision   = img_stats.get("from_vision", 0)
    skipped  = img_stats.get("skipped",    0)

    st.markdown("**🖼️ Image Processing Results:**")
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("⚡ Cached",     cached,   help="Free, from disk cache")
    c2.metric("🔍 OCR",        from_ocr, help="Free, Tesseract local OCR")
    c3.metric("🤖 Vision API", vision,   help="Gemini API calls used")
    c4.metric("⏭️ Skipped",    skipped,  help="Too small or duplicate")

    free = cached + from_ocr + skipped
    if total > 0:
        pct = int((free / total) * 100)
        st.info(
            f"💰 **{pct}%** of images processed without API calls "
            f"({free} / {total} free)"
        )


def render_qa_section(service: RAGService) -> None:
    """Render the question input and answer display panel."""
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

    col_btn, col_clear = st.columns([3, 1])

    with col_btn:
        search_clicked = st.button(
            "🔍 Get Answer",
            type="primary",
            use_container_width=True,
        )

    with col_clear:
        if st.button("🗑️ Clear", use_container_width=True):
            st.session_state["query"] = ""
            st.rerun()

    if search_clicked:
        _handle_query(service, user_query)


def _handle_query(service: RAGService, query: str) -> None:
    """Run the full RAG pipeline and render the response."""
    if not query.strip():
        st.warning("Please enter a question.")
        return

    with st.spinner("🤔 Searching documents and generating answer…"):
        response = service.answer(query)

    st.session_state["query"] = query

    if response.error == "rate_limited":
        st.warning(f"⏳ {response.answer}")
        return

    display_answer(response)
    display_retrieved_context(response.results)

    if response.answer.startswith("*(Cached)*"):
        st.caption("⚡ This answer was served from cache, no API call used.")


def _render_example_questions() -> None:
    st.markdown("**Quick questions:**")
    examples = [
        "What is this document about?",
        "Summarize the key points.",
        "What data or numbers are mentioned?",
        "Describe any charts or images.",
        "What are the main conclusions?",
        "List all topics covered.",
    ]
    cols = st.columns(2)
    for i, q in enumerate(examples):
        if cols[i % 2].button(q, key=f"ex_{i}"):
            st.session_state["query"] = q
            st.rerun()

    st.markdown("---")


def _render_tips() -> None:
    st.markdown("### 💡 Tips")
    st.markdown(
        "- Upload **PDF, Word, PowerPoint, Excel**, or **image** files\n"
        "- Images inside documents are automatically extracted and read\n"
        "- **Tesseract OCR** reads text images for free, no API calls\n"
        "- Ask specific questions for the most accurate answers\n"
        "- Upload **multiple files** at once and query across all of them\n"
        "- Repeated questions are answered from **cache** instantly"
    )