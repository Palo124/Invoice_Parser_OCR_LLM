import logging
from pathlib import Path

import fitz

logger = logging.getLogger(__name__)


def _layout_text_from_blocks(page: fitz.Page) -> str:
    blocks = page.get_text("blocks")
    lines: list[str] = []
    previous_bottom: float | None = None

    for block in sorted(blocks, key=lambda item: (item[1], item[0])):
        if len(block) < 5:
            continue
        x0, y0, _x1, y1, text, *_rest = block
        cleaned = str(text).strip()
        if not cleaned:
            continue

        if previous_bottom is not None and y0 - previous_bottom > 18:
            gap_lines = int((y0 - previous_bottom) // 18) - 1
            if gap_lines > 0:
                lines.extend([""] * gap_lines)

        block_lines = cleaned.splitlines()
        if x0 > page.rect.width * 0.45 and lines and " | " not in lines[-1]:
            lines[-1] = f"{lines[-1]} | {block_lines[0]}"
            lines.extend(block_lines[1:])
        else:
            lines.extend(block_lines)

        previous_bottom = y1

    return "\n".join(lines)


def extract_text_from_pdf(pdf_path: Path) -> tuple[str, list[str]]:
    """Extract text from a PDF using PyMuPDF (digital text layer)."""
    page_texts: list[str] = []

    with fitz.open(pdf_path) as document:
        for page in document:
            page_text = _layout_text_from_blocks(page).strip()
            if not page_text:
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
