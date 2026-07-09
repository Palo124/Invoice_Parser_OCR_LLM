import logging
from pathlib import Path

import fitz

logger = logging.getLogger(__name__)


def extract_text_from_pdf(pdf_path: Path) -> tuple[str, list[str]]:
    """Extract text from a PDF using PyMuPDF (digital text layer)."""
    page_texts: list[str] = []

    with fitz.open(pdf_path) as document:
        for page in document:
            page_text = page.get_text("text").strip()
            page_texts.append(page_text)

    combined = "\n\n".join(text for text in page_texts if text)
    logger.info(
        "PyMuPDF extracted %s chars across %s pages from %s",
        len(combined),
        len(page_texts),
        pdf_path.name,
    )
    return combined, page_texts
