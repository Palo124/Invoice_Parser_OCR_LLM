import json
from datetime import datetime
from typing import Any

from app.models.invoice import Invoice
from app.schemas.invoice_data import InvoiceData


def _parse_json_dict(raw: str | None) -> dict[str, Any]:
    if not raw:
        return {}
    return json.loads(raw)


def _flatten_values(data: dict[str, Any], prefix: str = "") -> dict[str, Any]:
    flat: dict[str, Any] = {}
    for key, value in data.items():
        path = f"{prefix}.{key}" if prefix else key
        if isinstance(value, dict):
            flat.update(_flatten_values(value, path))
        elif isinstance(value, list):
            for index, item in enumerate(value):
                item_path = f"{path}.{index}"
                if isinstance(item, dict):
                    flat.update(_flatten_values(item, item_path))
                else:
                    flat[item_path] = item
        else:
            flat[path] = value
    return flat


def _normalize(value: Any) -> Any:
    if value is None:
        return None
    if isinstance(value, float):
        return round(value, 4)
    return value


def compute_field_corrections(
    previous: dict[str, Any],
    updated: dict[str, Any],
    existing_corrections: dict[str, Any],
) -> dict[str, Any]:
    previous_flat = _flatten_values(previous)
    updated_flat = _flatten_values(updated)
    corrections = dict(existing_corrections)

    all_paths = set(previous_flat) | set(updated_flat)
    for path in sorted(all_paths):
        old_value = _normalize(previous_flat.get(path))
        new_value = _normalize(updated_flat.get(path))
        if old_value == new_value:
            continue

        prior = corrections.get(path)
        original = prior["from"] if prior else old_value
        if _normalize(original) == new_value:
            corrections.pop(path, None)
            continue

        corrections[path] = {
            "from": original,
            "to": new_value,
            "corrected_at": datetime.utcnow().isoformat(),
        }

    return corrections


def apply_invoice_corrections(invoice: Invoice, data: dict[str, Any]) -> None:
    if invoice.status != "completed":
        raise ValueError("Only completed invoices can be corrected")

    validated = InvoiceData.model_validate(data)
    updated = validated.to_storage_dict()
    previous = _parse_json_dict(invoice.data_json)
    existing = _parse_json_dict(invoice.corrected_fields_json)

    corrections = compute_field_corrections(previous, updated, existing)
    invoice.data_json = json.dumps(updated, ensure_ascii=False)
    invoice.corrected_fields_json = json.dumps(corrections, ensure_ascii=False) if corrections else None
    invoice.invoice_number = updated.get("invoice_number")
    supplier = updated.get("supplier") or {}
    invoice.supplier_name = supplier.get("name")

    if corrections:
        invoice.review_status = "corrected"
    elif invoice.review_status == "corrected":
        invoice.review_status = "pending" if invoice.needs_review else None


def approve_invoice(invoice: Invoice) -> None:
    if invoice.status != "completed":
        raise ValueError("Only completed invoices can be approved")
    if not invoice.data_json:
        raise ValueError("Invoice has no extracted data to approve")

    invoice.review_status = "approved"
    invoice.reviewed_at = datetime.utcnow()
    invoice.needs_review = False
