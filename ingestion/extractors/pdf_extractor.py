import io
from typing import List
from core.models import DocumentChunk, FileType
from .base_extractor import BaseExtractor

class PDFExtractor(BaseExtractor):
    SUPPORTED_EXTENSIONS = ["pdf"]

    def extract(self, file_path: str) -> List[DocumentChunk]:
        chunks: List[DocumentChunk] = []
        source = file_path.split("/")[-1]
        chunk_idx = 0

        try:
            import pdfplumber
            import fitz  # PyMuPDF

            with pdfplumber.open(file_path) as pdf:
                for page_num, page in enumerate(pdf.pages, start=1):
                    text = page.extract_text() or ""
                    table_text = self._extract_tables(page)
                    full_text = "\n".join(filter(None, [text, table_text])).strip()

                    if full_text:
                        chunks.append(self._make_text_chunk(
                            text=f"[Page {page_num}]\n{full_text}",
                            source=source,
                            page=page_num,
                            chunk_index=chunk_idx,
                            file_type=FileType.PDF,
                        ))
                        chunk_idx += 1

            doc = fitz.open(file_path)
            for page_num in range(len(doc)):
                page = doc[page_num]
                for img_idx, img in enumerate(page.get_images(full=True)):
                    xref = img[0]
                    base_image = doc.extract_image(xref)
                    image_bytes = base_image["image"]
                    ext = base_image["ext"]
                    mime = f"image/{ext}" if ext != "jpg" else "image/jpeg"

                    chunks.append(self._make_image_chunk(
                        image_data=image_bytes,
                        mime=mime,
                        source=source,
                        page=page_num + 1,
                        chunk_index=chunk_idx,
                        file_type=FileType.PDF,
                    ))
                    chunk_idx += 1

            doc.close()

        except ImportError as e:
            raise RuntimeError(
                f"Missing dependency for PDF extraction: {e}. "
                "Run: pip install pdfplumber pymupdf"
            )
        except Exception as e:
            raise RuntimeError(f"PDF extraction failed for '{file_path}': {e}")

        return chunks

    @staticmethod
    def _extract_tables(page) -> str:
        try:
            tables = page.extract_tables()
            if not tables:
                return ""
            rows = []
            for table in tables:
                for row in table:
                    clean_row = [cell or "" for cell in row]
                    rows.append(" | ".join(clean_row))
            return "\n".join(rows)
        except Exception:
            return ""