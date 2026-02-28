"""
OCREngine: extracts text from images using Tesseract OCR.
Completely free — no API calls. Used before Gemini Vision as first attempt.
Following SRP — only OCR logic lives here.
"""

import io
from typing import Optional, Tuple
from enum import Enum, auto

from config import config


class OCRResult(Enum):
    SUCCESS = auto()       # OCR found meaningful text
    LOW_CONFIDENCE = auto() # OCR ran but text too short/noisy
    FAILED = auto()        # OCR could not process image


class OCREngine:
    """
    Wraps Tesseract OCR with preprocessing for better accuracy.
    
    Pipeline per image:
      raw bytes → preprocess (denoise, contrast) → OCR → clean text → result
    """

    def __init__(self) -> None:
        self._available = self._setup_tesseract()

    @property
    def is_available(self) -> bool:
        return self._available

    def extract_text(self, image_data: bytes, mime: str = "image/png") -> Tuple[OCRResult, str]:
        """
        Run OCR on raw image bytes.
        Returns (OCRResult, extracted_text).
        """
        if not self._available:
            return OCRResult.FAILED, ""

        try:
            image = self._load_image(image_data)
            if image is None:
                return OCRResult.FAILED, ""

            # Preprocess for better OCR accuracy
            processed = self._preprocess(image)

            # Run OCR with multiple configs and pick best result
            text = self._run_ocr(processed)
            clean = self._clean_text(text)

            if len(clean) >= config.ocr.min_text_length:
                return OCRResult.SUCCESS, clean
            elif clean:
                return OCRResult.LOW_CONFIDENCE, clean
            else:
                return OCRResult.FAILED, ""

        except Exception:
            return OCRResult.FAILED, ""

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _setup_tesseract() -> bool:
        """Configure tesseract path and verify it's available."""
        try:
            import pytesseract
            if config.ocr.tesseract_path:
                pytesseract.pytesseract.tesseract_cmd = config.ocr.tesseract_path
            # Quick availability check
            pytesseract.get_tesseract_version()
            return True
        except Exception:
            return False

    @staticmethod
    def _load_image(image_data: bytes):
        """Load bytes into a PIL Image."""
        try:
            from PIL import Image
            return Image.open(io.BytesIO(image_data)).convert("RGB")
        except Exception:
            return None

    @staticmethod
    def _preprocess(image):
        """
        Preprocess image to improve OCR accuracy:
        - Upscale small images
        - Convert to grayscale
        - Increase contrast
        - Denoise
        """
        from PIL import Image, ImageFilter, ImageEnhance
        import io

        # Upscale if too small — Tesseract works better on larger images
        w, h = image.size
        if w < 300 or h < 300:
            scale = max(300 / w, 300 / h, 2.0)
            image = image.resize(
                (int(w * scale), int(h * scale)),
                Image.LANCZOS
            )

        # Grayscale
        image = image.convert("L")

        # Increase contrast
        image = ImageEnhance.Contrast(image).enhance(2.0)

        # Sharpen
        image = image.filter(ImageFilter.SHARPEN)

        return image

    @staticmethod
    def _run_ocr(image) -> str:
        """Run Tesseract with optimal config for document text."""
        import pytesseract

        # PSM 6 = assume uniform block of text (good for documents)
        # PSM 3 = fully automatic page segmentation (good for mixed content)
        configs = [
            "--psm 6 --oem 3",   # block of text
            "--psm 3 --oem 3",   # auto segmentation
            "--psm 11 --oem 3",  # sparse text (good for labels/captions)
        ]

        best_text = ""
        for cfg in configs:
            try:
                text = pytesseract.image_to_string(image, config=cfg)
                if len(text.strip()) > len(best_text.strip()):
                    best_text = text
            except Exception:
                continue

        return best_text

    @staticmethod
    def _clean_text(text: str) -> str:
        """Remove noise from OCR output."""
        if not text:
            return ""

        lines = text.splitlines()
        cleaned = []
        for line in lines:
            line = line.strip()
            # Skip lines that are just symbols/noise
            alpha_ratio = sum(c.isalnum() or c.isspace() for c in line) / max(len(line), 1)
            if line and alpha_ratio > 0.5:
                cleaned.append(line)

        return "\n".join(cleaned).strip()