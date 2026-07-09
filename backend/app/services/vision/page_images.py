from pathlib import Path

import numpy as np
from PIL import Image

from app.config import settings
from app.services.preprocessing.deskew import ImageDeskewer
from app.services.preprocessing.pdf_converter import PDFToImageConverter


def _deskew_page(page: np.ndarray) -> np.ndarray:
    try:
        deskewer = ImageDeskewer(page)
        rotated_page, _ = deskewer.deskew()
        return rotated_page
    except ValueError:
        return page


def load_page_images(file_path: Path) -> list[np.ndarray]:
    ext = file_path.suffix.lower()
    if ext == ".pdf":
        converter = PDFToImageConverter(poppler_path=None)
        pages = converter.convert_pdf_to_images(str(file_path), dpi=settings.ocr_pdf_dpi)
        return [np.array(page.convert("RGB")) for page in pages]

    if ext in {".png", ".jpg", ".jpeg"}:
        with Image.open(file_path) as image:
            page = np.array(image.convert("RGB"))
        return [_deskew_page(page)]

    raise ValueError("Unsupported file format. Use PDF, PNG, or JPEG.")
