VISION_PREFERRED_FIELDS = frozenset(
    {
        "items",
        "net_total",
        "tax_total",
        "gross_total",
    }
)

TEXT_PREFERRED_FIELDS = frozenset(
    {
        "invoice_number",
        "invoice_date",
        "tax_date",
        "due_date",
        "variable_symbol",
        "specific_symbol",
        "iban",
        "swift",
        "notes",
        "contact",
        "payment_method",
        "supplier",
        "customer",
        "original_filename",
        "source",
    }
)


def _has_value(value) -> bool:
    if value is None:
        return False
    if isinstance(value, str):
        return bool(value.strip())
    if isinstance(value, list):
        return len(value) > 0
    if isinstance(value, dict):
        return bool(value)
    return True


def _merge_party(text_party: dict | None, vision_party: dict | None) -> dict | None:
    text_party = text_party if isinstance(text_party, dict) else {}
    vision_party = vision_party if isinstance(vision_party, dict) else {}
    merged = dict(text_party)

    for field in ("name", "address", "ico", "dic", "platce_dph"):
        text_value = text_party.get(field)
        vision_value = vision_party.get(field)
        if field in {"name", "address"}:
            if _has_value(text_value):
                merged[field] = text_value
            elif _has_value(vision_value):
                merged[field] = vision_value
        elif _has_value(vision_value) and not _has_value(text_value):
            merged[field] = vision_value

    return merged or None


def merge_text_and_vision(text_data: dict, vision_data: dict) -> tuple[dict, list[str]]:
    merged = dict(text_data)
    merged_fields: list[str] = []

    for field in VISION_PREFERRED_FIELDS:
        vision_value = vision_data.get(field)
        if not _has_value(vision_value):
            continue
        if merged.get(field) != vision_value:
            merged[field] = vision_value
            merged_fields.append(field)

    for field in TEXT_PREFERRED_FIELDS:
        if field in {"supplier", "customer"}:
            party_merge = _merge_party(text_data.get(field), vision_data.get(field))
            if party_merge != text_data.get(field):
                merged[field] = party_merge
                if party_merge is not None:
                    merged_fields.append(field)
            continue

        text_value = text_data.get(field)
        vision_value = vision_data.get(field)
        if _has_value(text_value):
            merged[field] = text_value
        elif _has_value(vision_value):
            merged[field] = vision_value
            merged_fields.append(field)

    return merged, merged_fields
