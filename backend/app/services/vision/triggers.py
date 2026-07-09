import re

from app.config import settings
from app.services.text_extraction.ocr_comparator import OCRComparisonResult
from app.services.validation.rules import REQUIRED_FIELDS, _get_nested, _is_blank

_NUMERIC_LINE_RE = re.compile(r"(?:\d+[.,]\d+|\d+)")


def _has_many_numeric_columns(text: str) -> bool:
    numeric_lines = 0
    for line in text.splitlines():
        numbers = _NUMERIC_LINE_RE.findall(line)
        if len(numbers) >= settings.vision_min_numeric_columns:
            numeric_lines += 1
    return numeric_lines >= settings.vision_min_numeric_lines


def should_use_vision(
    *,
    branch: str,
    raw_text: str,
    page_count: int,
    text_data: dict,
    ocr_comparison: OCRComparisonResult | None = None,
) -> tuple[bool, list[str]]:
    if branch == "digital_pdf":
        return False, []

    reasons: list[str] = []

    if ocr_comparison is not None and ocr_comparison.agreement == "low":
        reasons.append("ocr_agreement_low")

    for field_path in REQUIRED_FIELDS:
        if _is_blank(_get_nested(text_data, field_path)):
            reasons.append(f"missing_{field_path.replace('.', '_')}")

    chars_per_page = len(raw_text) / max(page_count, 1)
    if chars_per_page < settings.vision_min_chars_per_page:
        reasons.append("low_text_density")

    if page_count > 1:
        reasons.append("multi_page")

    if _has_many_numeric_columns(raw_text):
        reasons.append("numeric_table_layout")

    return bool(reasons), reasons
