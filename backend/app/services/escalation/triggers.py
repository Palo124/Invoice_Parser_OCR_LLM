from app.config import settings
from app.services.escalation.disagreement import detect_text_vision_disagreements
from app.services.validation.types import ValidationResult

TOTALS_ID_CODES = frozenset(
    {
        "totals_mismatch",
        "invalid_ico",
        "invalid_ico_checksum",
        "invalid_party_id",
    }
)


def should_escalate(
    validation: ValidationResult,
    *,
    text_data: dict,
    vision_data: dict | None = None,
) -> tuple[bool, list[str]]:
    if not settings.escalation_enabled or settings.escalation_max_retries < 1:
        return False, []

    reasons: list[str] = []

    if validation.confidence == "low":
        reasons.append("confidence_low")

    error_codes = {error.code for error in validation.errors}
    if error_codes.intersection(TOTALS_ID_CODES):
        reasons.append("totals_or_id_validation_failed")

    if vision_data is not None:
        disagreements = detect_text_vision_disagreements(text_data, vision_data)
        if disagreements:
            reasons.append("text_vision_disagreement")

    return bool(reasons), reasons
