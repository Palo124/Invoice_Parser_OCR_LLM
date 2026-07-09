from app.config import settings
from app.schemas.invoice_data import InvoiceData
from app.services.text_extraction.ocr_comparator import OCRComparisonResult
from app.services.validation.confidence import score_validation
from app.services.validation.llm_self_check import check_numbers_in_source
from app.services.validation.rules import run_rule_checks
from app.services.validation.types import ValidationError, ValidationResult, collect_flags


class InvoiceValidationService:
    def validate(
        self,
        data: dict,
        *,
        raw_text: str = "",
        ocr_comparison: OCRComparisonResult | None = None,
        extraction_failed: bool = False,
    ) -> ValidationResult:
        errors: list[ValidationError] = []
        errors.extend(self._validate_schema(data))
        errors.extend(run_rule_checks(data, totals_tolerance=settings.validation_totals_tolerance))
        errors.extend(check_numbers_in_source(data, raw_text))

        ocr_conflict = False
        if ocr_comparison is not None and ocr_comparison.agreement == "low":
            ocr_conflict = True
            errors.append(
                ValidationError(
                    field="raw_text",
                    code="ocr_conflict",
                    message=(
                        f"Tesseract and Paddle OCR disagree "
                        f"(similarity={ocr_comparison.similarity:.2f})."
                    ),
                    severity="error",
                )
            )

        flags = collect_flags(errors)
        result = score_validation(
            errors=errors,
            flags=flags,
            ocr_conflict=ocr_conflict,
            extraction_failed=extraction_failed,
        )
        result.metadata = {
            "error_count": sum(1 for error in errors if error.severity == "error"),
            "warning_count": sum(1 for error in errors if error.severity == "warning"),
            "ocr_similarity": ocr_comparison.similarity if ocr_comparison else None,
        }
        return result

    def _validate_schema(self, data: dict) -> list[ValidationError]:
        try:
            InvoiceData.model_validate(data)
            return []
        except Exception as exc:
            return [
                ValidationError(
                    field="data",
                    code="schema_invalid",
                    message=f"pydantic_validation: {exc}",
                    severity="warning",
                )
            ]
