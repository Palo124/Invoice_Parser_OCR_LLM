import json


def normalize_value(value):
    if isinstance(value, (list, dict)):
        try:
            return json.dumps(value, sort_keys=True, ensure_ascii=False)
        except Exception:
            return str(value)
    return value


KEY_DISAGREEMENT_FIELDS = (
    "invoice_number",
    "gross_total",
    "net_total",
    "tax_total",
    "supplier.name",
    "supplier.ico",
    "items",
)


def _get_nested(data: dict, path: str):
    current = data
    for part in path.split("."):
        if not isinstance(current, dict):
            return None
        current = current.get(part)
    return current


def detect_text_vision_disagreements(text_data: dict, vision_data: dict) -> list[str]:
    disagreements: list[str] = []

    for field_path in KEY_DISAGREEMENT_FIELDS:
        left = _get_nested(text_data, field_path)
        right = _get_nested(vision_data, field_path)
        if normalize_value(left) != normalize_value(right):
            disagreements.append(field_path)

    return disagreements
