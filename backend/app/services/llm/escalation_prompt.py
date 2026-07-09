import json

from app.schemas.invoice_data import InvoiceData


def build_escalation_prompt(
    filename: str,
    source_text: str,
    current_json: dict,
    *,
    vision_json: dict | None = None,
    validation_errors: list[dict] | None = None,
    triggers: list[str] | None = None,
) -> str:
    schema_json = json.dumps(InvoiceData.model_json_schema(), indent=2, ensure_ascii=False)
    current_json_text = json.dumps(current_json, indent=2, ensure_ascii=False)
    vision_json_text = (
        json.dumps(vision_json, indent=2, ensure_ascii=False) if vision_json else "null"
    )
    errors_json = json.dumps(validation_errors or [], indent=2, ensure_ascii=False)
    triggers_text = ", ".join(triggers or []) or "none"

    return f"""
Předchozí extrakce faktury selhala validaci nebo je nekonzistentní. Oprav ji a vrať jeden JSON objekt podle schématu.
Zaměř se na označené problémy (totals, IČO, klíčová pole). Použij zdrojový text jako primární důkaz, vision JSON pro tabulky.

ESCALATION_TRIGGERS: {triggers_text}

VALIDATION_ERRORS:
{errors_json}

CURRENT_JSON:
{current_json_text}

VISION_JSON:
{vision_json_text}

JSON_SCHEMA:
{schema_json}

original_filename: {filename}

SOURCE_TEXT:
{source_text}
"""
