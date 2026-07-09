import logging
import tempfile
from pathlib import Path

import ocrmypdf

from app.services.text_extraction.pymupdf_extractor import extract_text_from_pdf

logger = logging.getLogger(__name__)


def extract_text_with_ocrmypdf(pdf_path: Path, language: str) -> tuple[str, list[str]]:
    """Run OCRmyPDF (Tesseract) on a PDF and extract text with PyMuPDF."""
    with tempfile.NamedTemporaryFile(suffix=".pdf", delete=False) as handle:
        output_path = Path(handle.name)

    try:
        logger.info("Running OCRmyPDF on %s (lang=%s)", pdf_path.name, language)
        ocrmypdf.ocr(
            pdf_path,
            output_path,
            language=language,
            skip_text=True,
            progress_bar=False,
        )
        return extract_text_from_pdf(output_path)
    finally:
        output_path.unlink(missing_ok=True)
