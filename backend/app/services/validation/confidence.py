from app.services.validation.types import ValidationError, ValidationResult


def score_validation(
    *,
    errors: list[ValidationError],
    flags: list[str],
    ocr_conflict: bool,
    extraction_failed: bool = False,
) -> ValidationResult:
    error_count = sum(1 for error in errors if error.severity == "error")
    warning_count = sum(1 for error in errors if error.severity == "warning")

    if extraction_failed or error_count >= 3:
        confidence = "failed"
    elif (
        ocr_conflict
        or "totals_mismatch" in flags
        or "tmr_disagreement" in flags
        or "required_field_missing" in flags
        or error_count > 0
    ):
        confidence = "low"
    elif warning_count > 0 or "llm_number_mismatch" in flags:
        confidence = "medium"
    else:
        confidence = "high"

    needs_review = (
        ocr_conflict
        or "totals_mismatch" in flags
        or "tmr_disagreement" in flags
        or confidence in {"low", "failed"}
    )

    review_status = "pending" if needs_review else None

    return ValidationResult(
        confidence=confidence,
        needs_review=needs_review,
        flags=flags,
        errors=errors,
        review_status=review_status,
    )
