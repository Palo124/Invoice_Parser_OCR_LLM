from app.services.types import ExtractionResult, PipelineResult, TextExtractionResult
from app.services.validation.types import ValidationResult


def finalize_pipeline_result(
    *,
    data: dict,
    extraction_path: str,
    validation: ValidationResult,
    steps: dict,
    metadata: dict,
    text_extraction: TextExtractionResult | None,
    extractions: list[ExtractionResult],
    raw_text: str,
    llm_raw_json: str,
    model_used: str,
) -> PipelineResult:
    steps["validation"] = validation.to_dict()
    return PipelineResult(
        data=data,
        extraction_path=extraction_path,
        confidence=validation.confidence,
        needs_review=validation.needs_review,
        flags=validation.flags,
        metadata={**metadata, "steps": steps},
        text_extraction=text_extraction,
        extractions=extractions,
        raw_text=raw_text,
        llm_raw_json=llm_raw_json,
        model_used=model_used,
        validation_errors=[error.to_dict() for error in validation.errors],
        review_status=validation.review_status,
    )
