import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image

from app.config import settings
from app.services.ocr.paddle_ocr_processor import PaddleOCRProcessor
from app.services.ocr.pytesseract_ocr_processor import PytesseractOCRProcessor
from app.services.preprocessing.deskew import ImageDeskewer
from app.services.preprocessing.pdf_converter import PDFToImageConverter
from app.services.text_extraction.ocr_comparator import OCRComparisonResult, compare_ocr_texts
from app.services.text_extraction.ocrmypdf_tesseract import extract_text_with_ocrmypdf
from app.services.text_extraction.pymupdf_extractor import extract_text_from_pdf
from app.services.text_extraction.text_quality import has_usable_text

logger = logging.getLogger(__name__)


@dataclass
class TextExtractionBundle:
    text: str
    source: str
    branch: str
    extraction_path: str
    page_count: int
    page_texts: list[str] = field(default_factory=list)
    confidence: float = 1.0
    ocr_comparison: OCRComparisonResult | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


class TextExtractionService:
    def __init__(self):
        self.tesseract = PytesseractOCRProcessor(
            tesseract_cmd=settings.tesseract_cmd,
            lang=settings.ocr_tesseract_lang,
        )
        self.paddle = PaddleOCRProcessor(
            lang=settings.ocr_paddle_lang,
            use_gpu=settings.ocr_paddle_gpu,
        )

    def extract(self, file_path: Path) -> TextExtractionBundle:
        ext = file_path.suffix.lower()
        if ext == ".pdf":
            return self._extract_pdf(file_path)
        if ext in {".png", ".jpg", ".jpeg"}:
            return self._extract_image(file_path)
        raise ValueError("Unsupported file format. Use PDF, PNG, or JPEG.")

    def _extract_pdf(self, file_path: Path) -> TextExtractionBundle:
        pymupdf_text, pymupdf_pages = extract_text_from_pdf(file_path)
        usable = has_usable_text(
            pymupdf_text,
            min_chars=settings.text_min_chars,
            max_garbage_ratio=settings.text_max_garbage_ratio,
            require_czech_signal=settings.text_require_czech_signal,
        )

        if usable:
            logger.info("Text extraction branch: digital_pdf (PyMuPDF) for %s", file_path.name)
            return TextExtractionBundle(
                text=pymupdf_text,
                source="pymupdf",
                branch="digital_pdf",
                extraction_path="modern:pymupdf_digital",
                page_count=len(pymupdf_pages),
                page_texts=pymupdf_pages,
                confidence=1.0,
                metadata={
                    "text_branch": "digital_pdf",
                    "pymupdf_chars": len(pymupdf_text),
                    "ocr_skipped": True,
                },
            )

        logger.info(
            "PyMuPDF text not usable for %s (%s chars); falling back to OCR path",
            file_path.name,
            len(pymupdf_text),
        )
        return self._extract_pdf_via_ocr(file_path, pymupdf_chars=len(pymupdf_text))

    def _extract_pdf_via_ocr(self, file_path: Path, pymupdf_chars: int = 0) -> TextExtractionBundle:
        tesseract_text, tesseract_pages = extract_text_with_ocrmypdf(
            file_path,
            settings.ocrmypdf_language,
        )
        page_images = self._load_pdf_images(file_path)
        paddle_text = self._paddle_ocr_pages(page_images)

        comparison = compare_ocr_texts(
            tesseract_text,
            paddle_text,
            agreement_threshold=settings.ocr_agreement_threshold,
        )
        primary_text = tesseract_text if len(tesseract_text) >= len(paddle_text) else paddle_text

        logger.info(
            "Text extraction branch: ocr_scan for %s (agreement=%s, similarity=%.3f)",
            file_path.name,
            comparison.agreement,
            comparison.similarity,
        )

        return TextExtractionBundle(
            text=primary_text,
            source="ocrmypdf+tesseract+paddle",
            branch="ocr_scan",
            extraction_path="modern:ocr_tesseract_paddle",
            page_count=len(page_images),
            page_texts=tesseract_pages,
            confidence=comparison.similarity,
            ocr_comparison=comparison,
            metadata={
                "text_branch": "ocr_scan",
                "pymupdf_chars_before_ocr": pymupdf_chars,
                "ocr_skipped": False,
                "tesseract_chars": len(tesseract_text),
                "paddle_chars": len(paddle_text),
                "ocr_agreement": comparison.agreement,
                "ocr_similarity": comparison.similarity,
            },
        )

    def _extract_image(self, file_path: Path) -> TextExtractionBundle:
        with Image.open(file_path) as image:
            pages = [np.array(image.convert("RGB"))]

        deskewed_pages = self._deskew_pages(pages)
        tesseract_text = self._tesseract_ocr_pages(deskewed_pages)
        paddle_text = self._paddle_ocr_pages(deskewed_pages)

        comparison = compare_ocr_texts(
            tesseract_text,
            paddle_text,
            agreement_threshold=settings.ocr_agreement_threshold,
        )
        primary_text = tesseract_text if len(tesseract_text) >= len(paddle_text) else paddle_text

        logger.info(
            "Text extraction branch: ocr_image for %s (agreement=%s, similarity=%.3f)",
            file_path.name,
            comparison.agreement,
            comparison.similarity,
        )

        return TextExtractionBundle(
            text=primary_text,
            source="tesseract+paddle",
            branch="ocr_image",
            extraction_path="modern:ocr_tesseract_paddle",
            page_count=1,
            page_texts=[tesseract_text],
            confidence=comparison.similarity,
            ocr_comparison=comparison,
            metadata={
                "text_branch": "ocr_image",
                "ocr_skipped": False,
                "tesseract_chars": len(tesseract_text),
                "paddle_chars": len(paddle_text),
                "ocr_agreement": comparison.agreement,
                "ocr_similarity": comparison.similarity,
            },
        )

    def _load_pdf_images(self, file_path: Path) -> list[np.ndarray]:
        converter = PDFToImageConverter(poppler_path=None)
        pages = converter.convert_pdf_to_images(str(file_path), dpi=settings.ocr_pdf_dpi)
        return [np.array(page.convert("RGB")) for page in pages]

    def _deskew_pages(self, pages: list[np.ndarray]) -> list[np.ndarray]:
        deskewed: list[np.ndarray] = []
        for page in pages:
            try:
                deskewer = ImageDeskewer(page)
                rotated_page, _ = deskewer.deskew()
            except ValueError:
                rotated_page = page
            deskewed.append(rotated_page)
        return deskewed

    def _tesseract_ocr_pages(self, pages: list[np.ndarray]) -> str:
        page_texts: list[str] = []
        for page in pages:
            pil_image = Image.fromarray(page)
            page_texts.append(
                self.tesseract.extract_text_layout_from_pil(
                    pil_image,
                    threshold=settings.ocr_tesseract_threshold,
                )
            )
        return "\n".join(page_texts)

    def _paddle_ocr_pages(self, pages: list[np.ndarray]) -> str:
        page_texts: list[str] = []
        for page in pages:
            pil_image = Image.fromarray(page)
            page_texts.append(
                self.paddle.extract_text_layout_from_pil(
                    pil_image,
                    threshold=settings.ocr_paddle_threshold,
                )
            )
        return "\n".join(page_texts)
