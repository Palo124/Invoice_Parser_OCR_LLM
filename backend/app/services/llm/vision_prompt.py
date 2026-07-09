import json

from app.schemas.invoice_data import InvoiceData


def get_vision_prompt(filename: str, page_count: int) -> str:
    schema_json = json.dumps(InvoiceData.model_json_schema(), indent=2, ensure_ascii=False)
    return f"""
Analyzuj přiložené stránky faktury ({page_count} obrázek/obrázků) a vrať jeden JSON objekt podle schématu.
Zaměř se zejména na tabulku položek, částky, DPH a součty.
Pro hlavičkové údaje (číslo faktury, dodavatel, datumy) použij text z dokumentu, pokud je čitelný.
Pokud údaj chybí nebo není čitelný, nastav hodnotu null.
Zachovej snake_case názvy polí.

JSON_SCHEMA:
{schema_json}

original_filename: {filename}
"""
