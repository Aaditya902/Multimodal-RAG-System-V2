"""
UI helper utilities — pure display functions, no business logic.
Following SRP: each function does exactly one UI thing.
"""

import base64
from typing import List

import streamlit as st

from core.models import QAResponse, RetrievalResult, FileType


def confidence_badge(confidence: float) -> str:
    """Return a colored badge string for the confidence score."""
    if confidence > 0.5:
        return f":green[High ({confidence:.2f})]"
    elif confidence > 0.3:
        return f":orange[Medium ({confidence:.2f})]"
    return f":red[Low ({confidence:.2f})]"


def file_type_icon(file_type: FileType) -> str:
    """Return an emoji icon for a file type."""
    icons = {
        FileType.PDF: "📄",
        FileType.WORD: "📝",
        FileType.POWERPOINT: "📊",
        FileType.EXCEL: "📈",
        FileType.IMAGE: "🖼️",
    }
    return icons.get(file_type, "📁")


def display_answer(response: QAResponse) -> None:
    """Render the QA response in the UI."""
    st.markdown("### 📝 Answer")
    if response.error:
        st.error(f"⚠️ Generation error: {response.error}")
    st.markdown(response.answer)
    st.markdown(f"**Confidence:** {confidence_badge(response.confidence)}")


def display_retrieved_context(results: List[RetrievalResult], max_text_len: int = 400) -> None:
    """Show the retrieved context chunks in an expander."""
    label = f"🔍 Retrieved Context ({len(results)} chunk{'s' if len(results) != 1 else ''} found)"

    with st.expander(label):
        if not results:
            st.info("No relevant chunks found. Try rephrasing your question.")
            return

        for i, result in enumerate(results, start=1):
            chunk = result.chunk
            icon = file_type_icon(chunk.file_type)
            st.markdown(
                f"**{icon} Chunk {i}** — `{chunk.source}` "
                f"| Page {chunk.page} | Similarity: `{result.similarity:.2f}`"
            )

            if chunk.is_image:
                if chunk.image_data:
                    b64 = base64.b64encode(chunk.image_data).decode("utf-8")
                    mime = chunk.image_mime or "image/png"
                    st.markdown(
                        f'<img src="data:{mime};base64,{b64}" style="max-width:400px;border-radius:8px"/>',
                        unsafe_allow_html=True,
                    )
                if chunk.image_description:
                    st.caption(f"🤖 Description: {chunk.image_description[:300]}")
            else:
                preview = chunk.text[:max_text_len]
                if len(chunk.text) > max_text_len:
                    preview += "…"
                st.markdown(preview)

            st.markdown("---")


def display_indexed_summary(chunks) -> None:
    """Show a compact summary of what's been indexed."""
    from collections import Counter
    type_counts = Counter(c.file_type for c in chunks)
    image_count = sum(1 for c in chunks if c.is_image)
    text_count = len(chunks) - image_count

    cols = st.columns(3)
    cols[0].metric("Total Chunks", len(chunks))
    cols[1].metric("Text Chunks", text_count)
    cols[2].metric("Image Chunks", image_count)

    if type_counts:
        breakdown = ", ".join(
            f"{file_type_icon(ft)} {ft.name}: {n}" for ft, n in type_counts.items()
        )
        st.caption(f"By type: {breakdown}")


def cleanup_temp_file(file_path: str) -> None:
    """Safely delete a temporary file."""
    import os
    try:
        if os.path.exists(file_path):
            os.remove(file_path)
    except OSError:
        pass  # Best-effort cleanup