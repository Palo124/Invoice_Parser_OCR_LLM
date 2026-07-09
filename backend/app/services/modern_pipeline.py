from pathlib import Path

from app.config import settings
from app.services.exceptions import ProcessingCancelled
from app.services.llm.extractor import InvoiceLLMExtractor
from app.services.llm.vision_extractor import InvoiceVisionExtractor
from app.services.pipeline_common import finalize_pipeline_result
from app.services.text_extraction.service import TextExtractionService
from app.services.types import (
    CancelCheck,
    ExtractionResult,
    PipelineResult,
    ProgressCallback,
    TextExtractionResult,
)
from app.services.validation import InvoiceValidationService
from app.services.vision import merge_text_and_vision, should_use_vision
from app.services.vision.page_images import load_page_images


class ModernPipeline:
    """Phase 1–4: text extraction, LLM extraction, optional vision fallback, validation."""

    def __init__(self):
        self.text_extractor = TextExtractionService()
        self.llm_extractor = InvoiceLLMExtractor()
        self.vision_extractor = InvoiceVisionExtractor()
        self.validator = InvoiceValidationService()

    def process_file(
        self,
        file_path: Path,
        on_progress: ProgressCallback | None = None,
        should_cancel: CancelCheck | None = None,
    ) -> PipelineResult:
        filename = file_path.name
        steps: dict = {"text_extraction": {}, "ocr": [], "llm": []}

        def check_cancelled() -> None:
            if should_cancel and should_cancel():
                raise ProcessingCancelled()

        def report(stage: str) -> None:
            check_cancelled()
            if on_progress:
                on_progress(stage, steps)

        check_cancelled()
        bundle = self.text_extractor.extract(file_path)
        steps["text_extraction"] = {
            "branch": bundle.branch,
            "source": bundle.source,
            "extraction_path": bundle.extraction_path,
            "page_count": bundle.page_count,
            "char_count": len(bundle.text),
            "preview": bundle.text[:500] + ("…" if len(bundle.text) > 500 else ""),
            **bundle.metadata,
        }
        report("text:pymupdf")

        if bundle.ocr_comparison:
            steps["ocr"] = [
                {
                    "engine": "tesseract",
                    "text": bundle.ocr_comparison.tesseract_text,
                    "char_count": len(bundle.ocr_comparison.tesseract_text),
                    "preview": bundle.ocr_comparison.tesseract_text[:500],
                },
                {
                    "engine": "paddleocr",
                    "text": bundle.ocr_comparison.paddle_text,
                    "char_count": len(bundle.ocr_comparison.paddle_text),
                    "preview": bundle.ocr_comparison.paddle_text[:500],
                },
            ]
            steps["text_extraction"]["ocr_comparison"] = {
                "similarity": bundle.ocr_comparison.similarity,
                "agreement": bundle.ocr_comparison.agreement,
            }
            report("text:ocr_compare")
        else:
            report("text:ocr_skipped")

        text_extraction = TextExtractionResult(
            text=bundle.text,
            source=bundle.source,
            confidence=bundle.confidence,
        )

        check_cancelled()
        report("llm:deepseek")
        llm_result = self.llm_extractor.extract(filename, bundle.text, bundle.source)

        extractions = [
            ExtractionResult(
                data=llm_result.parsed_data,
                model=llm_result.model,
                raw_output=llm_result.raw_output,
                ocr_engine=bundle.source,
                prompt_tokens=llm_result.prompt_tokens,
                completion_tokens=llm_result.completion_tokens,
            )
        ]
        steps["llm"].append(
            {
                "model": llm_result.model,
                "ocr_engine": bundle.source,
                "raw_output": llm_result.raw_output,
                "parsed_json": llm_result.parsed_data,
                "prompt_tokens": llm_result.prompt_tokens,
                "completion_tokens": llm_result.completion_tokens,
                "structured_output": llm_result.structured_output,
            }
        )

        final_data = llm_result.parsed_data
        extraction_path = bundle.extraction_path
        total_tokens = llm_result.prompt_tokens + llm_result.completion_tokens
        total_cost = llm_result.estimated_cost
        models = [llm_result.model]
        vision_used = False
        vision_triggers: list[str] = []

        run_vision, vision_triggers = should_use_vision(
            branch=bundle.branch,
            raw_text=bundle.text,
            page_count=bundle.page_count,
            text_data=llm_result.parsed_data,
            ocr_comparison=bundle.ocr_comparison,
        )
        steps["vision_trigger"] = {
            "enabled": settings.vision_enabled,
            "should_run": run_vision,
            "reasons": vision_triggers,
        }

        if settings.vision_enabled and run_vision:
            check_cancelled()
            report("llm:vision")
            page_images = load_page_images(file_path)
            vision_result = self.vision_extractor.extract(filename, page_images)
            merged_data, merged_fields = merge_text_and_vision(
                llm_result.parsed_data,
                vision_result.parsed_data,
            )
            final_data = merged_data
            vision_used = True
            extraction_path = f"{bundle.extraction_path}+vision"
            total_tokens += vision_result.prompt_tokens + vision_result.completion_tokens
            total_cost += vision_result.estimated_cost
            models.append(vision_result.model)

            extractions.append(
                ExtractionResult(
                    data=vision_result.parsed_data,
                    model=vision_result.model,
                    raw_output=vision_result.raw_output,
                    ocr_engine="vision",
                    prompt_tokens=vision_result.prompt_tokens,
                    completion_tokens=vision_result.completion_tokens,
                )
            )
            steps["vision"] = {
                "model": vision_result.model,
                "page_count": vision_result.page_count,
                "raw_output": vision_result.raw_output,
                "parsed_json": vision_result.parsed_data,
                "prompt_tokens": vision_result.prompt_tokens,
                "completion_tokens": vision_result.completion_tokens,
                "triggers": vision_triggers,
            }
            steps["vision_merge"] = {
                "merged_fields": merged_fields,
                "merged_json": merged_data,
            }

        check_cancelled()
        report("validation")
        validation = self.validator.validate(
            final_data,
            raw_text=bundle.text,
            ocr_comparison=bundle.ocr_comparison,
        )

        return finalize_pipeline_result(
            data=final_data,
            extraction_path=extraction_path,
            validation=validation,
            steps=steps,
            metadata={
                "pipeline_mode": "modern",
                "text_branch": bundle.branch,
                "token_usage": {"total_tokens": total_tokens},
                "estimated_cost": total_cost,
                "models": models,
                "structured_output": llm_result.structured_output,
                "vision_used": vision_used,
                "vision_triggers": vision_triggers if vision_used else [],
                **bundle.metadata,
            },
            text_extraction=text_extraction,
            extractions=extractions,
            raw_text=bundle.text,
            llm_raw_json=llm_result.raw_output,
            model_used=",".join(models),
        )
