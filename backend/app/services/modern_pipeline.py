from pathlib import Path

from app.services.exceptions import ProcessingCancelled
from app.services.llm.extractor import InvoiceLLMExtractor
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


class ModernPipeline:
    """Phase 1–3: text extraction, single LLM extraction, validation layer."""

    def __init__(self):
        self.text_extractor = TextExtractionService()
        self.llm_extractor = InvoiceLLMExtractor()
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

        extraction = ExtractionResult(
            data=llm_result.parsed_data,
            model=llm_result.model,
            raw_output=llm_result.raw_output,
            ocr_engine=bundle.source,
            prompt_tokens=llm_result.prompt_tokens,
            completion_tokens=llm_result.completion_tokens,
        )
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

        check_cancelled()
        report("validation")
        validation = self.validator.validate(
            llm_result.parsed_data,
            raw_text=bundle.text,
            ocr_comparison=bundle.ocr_comparison,
        )

        return finalize_pipeline_result(
            data=llm_result.parsed_data,
            extraction_path=bundle.extraction_path,
            validation=validation,
            steps=steps,
            metadata={
                "pipeline_mode": "modern",
                "text_branch": bundle.branch,
                "token_usage": {
                    "total_tokens": llm_result.prompt_tokens + llm_result.completion_tokens,
                },
                "estimated_cost": llm_result.estimated_cost,
                "models": [llm_result.model],
                "structured_output": llm_result.structured_output,
                **bundle.metadata,
            },
            text_extraction=text_extraction,
            extractions=[extraction],
            raw_text=bundle.text,
            llm_raw_json=llm_result.raw_output,
            model_used=llm_result.model,
        )
