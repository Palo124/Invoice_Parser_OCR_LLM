# pytesseract_ocr_processor.py
import pytesseract
from pytesseract import Output
from PIL import Image

from app.config import settings
from app.services.ocr.layout_reconstruction import layout_text_from_items


class PytesseractOCRProcessor:
    def __init__(self, tesseract_cmd=None, lang="ces"):
        if tesseract_cmd:
            pytesseract.pytesseract.tesseract_cmd = tesseract_cmd
        self.lang = lang

    def extract_text(self, image_path):
        """Extract plain text (no layout preservation) given a file path."""
        image = Image.open(image_path)
        text = pytesseract.image_to_string(image, lang=self.lang)
        return text

    def extract_text_from_pil(self, pil_image):
        """Extract plain text (no layout preservation) from a PIL image."""
        text = pytesseract.image_to_string(pil_image, lang=self.lang)
        return text

    def _items_from_image(self, pil_image) -> list[dict]:
        data = pytesseract.image_to_data(pil_image, lang=self.lang, output_type=Output.DICT)
        items = []
        for index, text in enumerate(data["text"]):
            cleaned = text.strip()
            if not cleaned:
                continue
            left = data["left"][index]
            width = data["width"][index]
            items.append(
                {
                    "text": cleaned,
                    "x_min": left,
                    "y_min": data["top"][index],
                    "x_max": left + width,
                }
            )
        return items

    def extract_text_layout_from_pil(self, pil_image, threshold) -> str:
        items = self._items_from_image(pil_image)
        return layout_text_from_items(
            items,
            line_threshold=threshold,
            page_width=pil_image.size[0],
            column_split_enabled=settings.ocr_column_split_enabled,
            column_gap_ratio=settings.ocr_column_gap_ratio,
            char_width=settings.ocr_layout_char_width,
            space_divisor=settings.ocr_layout_space_divisor,
            blank_line_y_multiplier=settings.ocr_layout_blank_line_multiplier,
        )
