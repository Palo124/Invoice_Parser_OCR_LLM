from pathlib import Path
from typing import Any

from jinja2 import Environment, FileSystemLoader, select_autoescape

TEMPLATES_DIR = Path(__file__).resolve().parent / "templates"


def _display(value: Any, fallback: str = "—") -> str:
    if value is None:
        return fallback
    if isinstance(value, bool):
        return "Ano" if value else "Ne"
    text = str(value).strip()
    return text if text else fallback


def _party_block(data: dict | None) -> dict[str, Any]:
    data = data or {}
    return {
        "name": _display(data.get("name"), ""),
        "address": _display(data.get("address"), ""),
        "ico": _display(data.get("ico"), ""),
        "dic": _display(data.get("dic"), ""),
        "platce_dph": data.get("platce_dph"),
    }


def _normalize_items(items: list | None) -> list[dict[str, Any]]:
    if not items:
        return []

    normalized = []
    for index, item in enumerate(items, start=1):
        if not isinstance(item, dict):
            continue
        normalized.append(
            {
                "index": item.get("index") or index,
                "description": _display(item.get("description"), ""),
                "quantity": _display(item.get("quantity"), ""),
                "unit": _display(item.get("unit"), ""),
                "unit_price": _display(item.get("unit_price"), ""),
                "tax_rate": _display(item.get("tax_rate"), ""),
                "net_amount": _display(item.get("net_amount"), ""),
                "tax_amount": _display(item.get("tax_amount"), ""),
                "gross_amount": _display(item.get("gross_amount"), ""),
            }
        )
    return normalized


def build_invoice_context(data: dict[str, Any]) -> dict[str, Any]:
    supplier = _party_block(data.get("supplier"))
    customer = _party_block(data.get("customer"))

    return {
        "invoice_number": _display(data.get("invoice_number")),
        "invoice_date": _display(data.get("invoice_date")),
        "tax_date": _display(data.get("tax_date")),
        "due_date": _display(data.get("due_date")),
        "variable_symbol": _display(data.get("variable_symbol")),
        "specific_symbol": _display(data.get("specific_symbol")),
        "iban": _display(data.get("iban")),
        "swift": _display(data.get("swift")),
        "payment_method": _display(data.get("payment_method")),
        "net_total": _display(data.get("net_total")),
        "tax_total": _display(data.get("tax_total")),
        "gross_total": _display(data.get("gross_total")),
        "notes": _display(data.get("notes"), ""),
        "contact": _display(data.get("contact"), ""),
        "original_filename": _display(data.get("original_filename"), ""),
        "supplier": supplier,
        "customer": customer,
        "items": _normalize_items(data.get("items")),
    }


def render_invoice_html(data: dict[str, Any]) -> str:
    env = Environment(
        loader=FileSystemLoader(TEMPLATES_DIR),
        autoescape=select_autoescape(["html", "xml"]),
    )
    template = env.get_template("invoice.html")
    return template.render(**build_invoice_context(data))
