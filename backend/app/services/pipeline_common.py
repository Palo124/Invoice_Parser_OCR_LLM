import time

from app.services.types import ExtractionResult, PipelineResult, TextExtractionResult
from app.services.validation.types import ValidationResult


def round_duration(seconds: float) -> float:
    return round(seconds, 3)


def start_timer() -> float:
    return time.perf_counter()


def elapsed_seconds(started_at: float) -> float:
    return round_duration(time.perf_counter() - started_at)


def summarize_step_metrics(steps: dict) -> dict:
    total_duration = 0.0
    total_cost = 0.0
    by_stage: list[dict] = []

    text_extraction = steps.get("text_extraction")
    if isinstance(text_extraction, dict):
        duration = float(text_extraction.get("duration_seconds") or 0)
        cost = float(text_extraction.get("estimated_cost") or 0)
        total_duration += duration
        total_cost += cost
        by_stage.append(
            {
                "stage": "text_extraction",
                "duration_seconds": duration,
                "estimated_cost": cost,
            }
        )

    for llm_step in steps.get("llm") or []:
        duration = float(llm_step.get("duration_seconds") or 0)
        cost = float(llm_step.get("estimated_cost") or 0)
        total_duration += duration
        total_cost += cost
        by_stage.append(
            {
                "stage": "llm",
                "model": llm_step.get("model"),
                "duration_seconds": duration,
                "estimated_cost": cost,
            }
        )

    for stage_name in ("vision", "escalation", "validation"):
        step = steps.get(stage_name)
        if not isinstance(step, dict):
            continue
        duration = float(step.get("duration_seconds") or 0)
        cost = float(step.get("estimated_cost") or 0)
        total_duration += duration
        total_cost += cost
        entry = {
            "stage": stage_name,
            "duration_seconds": duration,
            "estimated_cost": cost,
        }
        if stage_name in {"vision", "escalation"}:
            entry["model"] = step.get("model")
        by_stage.append(entry)

    return {
        "total_duration_seconds": round_duration(total_duration),
        "estimated_cost": round(total_cost, 6),
        "by_stage": by_stage,
    }


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
    validation_payload = validation.to_dict()
    existing_validation = steps.get("validation")
    if isinstance(existing_validation, dict):
        validation_payload = {
            **validation_payload,
            "duration_seconds": existing_validation.get("duration_seconds"),
            "estimated_cost": existing_validation.get("estimated_cost", 0.0),
        }
    steps["validation"] = validation_payload
    metrics = summarize_step_metrics(steps)
    return PipelineResult(
        data=data,
        extraction_path=extraction_path,
        confidence=validation.confidence,
        needs_review=validation.needs_review,
        flags=validation.flags,
        metadata={
            **metadata,
            "steps": steps,
            "total_duration_seconds": metrics["total_duration_seconds"],
            "estimated_cost": metrics["estimated_cost"],
            "step_metrics": metrics["by_stage"],
        },
        text_extraction=text_extraction,
        extractions=extractions,
        raw_text=raw_text,
        llm_raw_json=llm_raw_json,
        model_used=model_used,
        validation_errors=[error.to_dict() for error in validation.errors],
        review_status=validation.review_status,
    )
