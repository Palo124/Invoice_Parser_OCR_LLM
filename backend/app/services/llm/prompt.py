import json

from app.schemas.invoice_data import InvoiceData


def get_invoice_json_schema() -> dict:
    return InvoiceData.model_json_schema()


def get_prompt(filename: str, source_text: str) -> str:
    schema_json = json.dumps(get_invoice_json_schema(), indent=2, ensure_ascii=False)
    return f"""
Níže je text faktury. Analyzuj ho a vrať jeden JSON objekt podle schématu.
Pokud údaj chybí nebo není čitelný, nastav hodnotu null.
Pro pole ico a dic: IČO = 8 číslic, DIČ začíná „CZ“ + 8 až 10 číslic.
Zachovej snake_case názvy polí.

JSON_SCHEMA:
{schema_json}

original_filename: {filename}

INVOICE_TEXT:
{source_text}
"""


def build_messages(filename: str, source_text: str) -> list[dict]:
    return [{"role": "user", "content": get_prompt(filename, source_text)}]
