import re
from typing import Any

from app.services.validation.types import ValidationError

_NUMBER_FIELDS = (
    "invoice_number",
    "variable_symbol",
    "net_total",
    "tax_total",
    "gross_total",
)


def _normalize_source(source: str) -> str:
    return source.lower()


def _number_variants(value: float | int | str) -> list[str]:
    if isinstance(value, str):
        cleaned = value.strip()
        if not cleaned:
            return []
        try:
            numeric = float(cleaned.replace(" ", "").replace(",", "."))
        except ValueError:
            return [cleaned.lower()]
        value = numeric

    if isinstance(value, int):
        value = float(value)

    variants = set()
    if isinstance(value, float):
        if value.is_integer():
            int_value = int(value)
            variants.update(
                {
                    str(int_value),
                    f"{int_value:,}".replace(",", " "),
                    f"{int_value:,}".replace(",", "."),
                }
            )
        variants.update(
            {
                f"{value:.2f}",
                f"{value:.2f}".replace(".", ","),
                str(value),
                str(value).replace(".", ","),
            }
        )
    return [variant.lower() for variant in variants if variant]


def _value_in_source(value: Any, source: str) -> bool:
    if value is None:
        return True
    if isinstance(value, str) and not value.strip():
        return True

    normalized_source = _normalize_source(source)
    if isinstance(value, str):
        token = value.strip().lower()
        return token in normalized_source

    for variant in _number_variants(value):
        if variant and variant in normalized_source:
            return True
        compact = re.sub(r"[\s.,]", "", variant)
        if compact and compact in re.sub(r"[\s.,]", "", normalized_source):
            return True
    return False


def check_numbers_in_source(data: dict, source_text: str) -> list[ValidationError]:
    errors: list[ValidationError] = []

    if not source_text.strip():
        return errors

    for field_name in _NUMBER_FIELDS:
        value = data.get(field_name)
        if value is None or (isinstance(value, str) and not value.strip()):
            continue
        if not _value_in_source(value, source_text):
            errors.append(
                ValidationError(
                    field=field_name,
                    code="llm_number_mismatch",
                    message=f"Extracted value for '{field_name}' was not found in source text.",
                    severity="warning",
                )
            )

    return errors
