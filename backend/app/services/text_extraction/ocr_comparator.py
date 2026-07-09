import re
from dataclasses import dataclass
from difflib import SequenceMatcher


@dataclass
class OCRComparisonResult:
    tesseract_text: str
    paddle_text: str
    similarity: float
    agreement: str


def _normalize_text(text: str) -> str:
    lowered = text.lower()
    lowered = re.sub(r"[^\w\s]", " ", lowered, flags=re.UNICODE)
    return re.sub(r"\s+", " ", lowered).strip()


def compare_ocr_texts(
    tesseract_text: str,
    paddle_text: str,
    *,
    agreement_threshold: float,
) -> OCRComparisonResult:
    left = _normalize_text(tesseract_text)
    right = _normalize_text(paddle_text)

    if not left and not right:
        similarity = 1.0
    elif not left or not right:
        similarity = 0.0
    else:
        similarity = SequenceMatcher(None, left, right).ratio()

    agreement = "high" if similarity >= agreement_threshold else "low"

    return OCRComparisonResult(
        tesseract_text=tesseract_text,
        paddle_text=paddle_text,
        similarity=round(similarity, 4),
        agreement=agreement,
    )
