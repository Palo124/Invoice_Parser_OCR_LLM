import re
from datetime import datetime

from app.services.validation.types import ValidationError

_ICO_RE = re.compile(r"^\d{8}$")
_DATE_FORMATS = ("%d.%m.%Y", "%Y-%m-%d", "%d/%m/%Y", "%d.%m.%y")

REQUIRED_FIELDS = (
    "invoice_number",
    "supplier.name",
    "gross_total",
)


def _get_nested(data: dict, path: str):
    current = data
    for part in path.split("."):
        if not isinstance(current, dict):
            return None
        current = current.get(part)
    return current


def _is_blank(value) -> bool:
    return value is None or (isinstance(value, str) and not value.strip())


def _ico_checksum_valid(ico: str) -> bool:
    if not _ICO_RE.match(ico):
        return False
    weights = (8, 7, 6, 5, 4, 3, 2)
    total = sum(int(digit) * weight for digit, weight in zip(ico[:7], weights, strict=True))
    remainder = total % 11
    if remainder == 0:
        check = 1
    elif remainder == 1:
        check = 0
    else:
        check = 11 - remainder
    if check == 10:
        check = 0
    return int(ico[7]) == check


def _parse_date(value: str) -> bool:
    stripped = value.strip()
    for fmt in _DATE_FORMATS:
        try:
            datetime.strptime(stripped, fmt)
            return True
        except ValueError:
            continue
    return False


def _float_or_none(value) -> float | None:
    if value is None:
        return None
    if isinstance(value, (int, float)):
        return float(value)
    if isinstance(value, str):
        cleaned = value.strip().replace(" ", "").replace(",", ".")
        try:
            return float(cleaned)
        except ValueError:
            return None
    return None


def _within_tolerance(expected: float, actual: float, tolerance: float) -> bool:
    return abs(expected - actual) <= tolerance


def check_required_fields(data: dict) -> list[ValidationError]:
    errors: list[ValidationError] = []

    for field_path in REQUIRED_FIELDS:
        value = _get_nested(data, field_path)
        if _is_blank(value):
            errors.append(
                ValidationError(
                    field=field_path,
                    code="required_field",
                    message=f"Required field '{field_path}' is missing.",
                    severity="error",
                )
            )
    return errors


def check_ico(data: dict) -> list[ValidationError]:
    errors: list[ValidationError] = []

    for party_key in ("supplier", "customer"):
        party = data.get(party_key) or {}
        if not isinstance(party, dict):
            continue

        ico = party.get("ico")
        if not _is_blank(ico):
            normalized = re.sub(r"\s+", "", str(ico))
            if not _ICO_RE.match(normalized):
                errors.append(
                    ValidationError(
                        field=f"{party_key}.ico",
                        code="invalid_ico",
                        message="IČO must be exactly 8 digits.",
                        severity="error",
                    )
                )
            elif not _ico_checksum_valid(normalized):
                errors.append(
                    ValidationError(
                        field=f"{party_key}.ico",
                        code="invalid_ico_checksum",
                        message="IČO checksum is invalid.",
                        severity="warning",
                    )
                )

    return errors


def check_dates(data: dict) -> list[ValidationError]:
    errors: list[ValidationError] = []

    for field_name in ("invoice_date", "tax_date", "due_date"):
        value = data.get(field_name)
        if _is_blank(value):
            continue
        if not _parse_date(str(value)):
            errors.append(
                ValidationError(
                    field=field_name,
                    code="invalid_date",
                    message=f"Could not parse date '{value}'.",
                    severity="warning",
                )
            )

    return errors


def check_totals(data: dict, *, tolerance: float) -> list[ValidationError]:
    errors: list[ValidationError] = []

    net_total = _float_or_none(data.get("net_total"))
    tax_total = _float_or_none(data.get("tax_total"))
    gross_total = _float_or_none(data.get("gross_total"))

    items = data.get("items") or []
    if isinstance(items, list) and items:
        items_net = sum(
            amount
            for item in items
            if isinstance(item, dict)
            for amount in [_float_or_none(item.get("net_amount"))]
            if amount is not None
        )
        items_gross = sum(
            amount
            for item in items
            if isinstance(item, dict)
            for amount in [_float_or_none(item.get("gross_amount"))]
            if amount is not None
        )

        if net_total is not None and items_net > 0 and not _within_tolerance(items_net, net_total, tolerance):
            message = f"Line items net sum ({items_net}) differs from net_total ({net_total})."
            errors.append(
                ValidationError(
                    field="net_total",
                    code="totals_mismatch",
                    message=message,
                    severity="error",
                )
            )

        if gross_total is not None and items_gross > 0 and not _within_tolerance(items_gross, gross_total, tolerance):
            message = f"Line items gross sum ({items_gross}) differs from gross_total ({gross_total})."
            errors.append(
                ValidationError(
                    field="gross_total",
                    code="totals_mismatch",
                    message=message,
                    severity="error",
                )
            )

    if net_total is not None and tax_total is not None and gross_total is not None:
        expected_gross = net_total + tax_total
        if not _within_tolerance(expected_gross, gross_total, tolerance):
            message = (
                f"net_total ({net_total}) + tax_total ({tax_total}) "
                f"does not match gross_total ({gross_total})."
            )
            errors.append(
                ValidationError(
                    field="gross_total",
                    code="totals_mismatch",
                    message=message,
                    severity="error",
                )
            )

    return errors


def run_rule_checks(data: dict, *, totals_tolerance: float) -> list[ValidationError]:
    errors: list[ValidationError] = []

    for check in (
        check_required_fields,
        check_ico,
        check_dates,
    ):
        errors.extend(check(data))

    errors.extend(check_totals(data, tolerance=totals_tolerance))
    return errors
