from pathlib import Path

from app.config import settings
from app.services.exceptions import ProcessingCancelled
from app.services.llm.deepinfra import DeepInfraClient
from app.services.llm.json_extractor import JSONExtractor
from app.services.llm.prompt import get_prompt
from app.services.text_extraction.service import TextExtractionService
from app.services.types import (
    CancelCheck,
    ExtractionResult,
    PipelineResult,
    ProgressCallback,
    TextExtractionResult,
)


class ModernPipeline:
    """Phase 1 pipeline: decision-tree text extraction + interim single-LLM extraction."""

    def __init__(self):
        if not settings.deepinfra_api_key:
            raise ValueError("DEEPINFRA_API_KEY is not set")

        self.text_extractor = TextExtractionService()
        self.llm = DeepInfraClient(
            settings.deepinfra_api_key,
            settings.llm_deepseek_model,
        )

    def _call_llm(self, filename: str, text: str):
        prompt = [{"role": "user", "content": get_prompt(filename, text)}]
        return self.llm.get_chat_completion(prompt, temperature=0.0)

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
        response = self._call_llm(filename, bundle.text)
        content = response.choices[0].message.content or ""
        parsed = JSONExtractor.extract_json(content)
        prompt_tokens = response.usage.prompt_tokens
        completion_tokens = response.usage.completion_tokens

        extraction = ExtractionResult(
            data=parsed,
            model=settings.llm_deepseek_model,
            confidence="high" if bundle.ocr_comparison is None else bundle.ocr_comparison.agreement,
            raw_output=content,
            ocr_engine=bundle.source,
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
        )
        steps["llm"].append(
            {
                "model": settings.llm_deepseek_model,
                "ocr_engine": bundle.source,
                "raw_output": content,
                "parsed_json": parsed,
                "prompt_tokens": prompt_tokens,
                "completion_tokens": completion_tokens,
            }
        )

        total_tokens = prompt_tokens + completion_tokens
        total_cost = getattr(response.usage, "estimated_cost", 0.0)
        needs_review = bundle.ocr_comparison is not None and bundle.ocr_comparison.agreement == "low"
        confidence = "high" if not needs_review else "medium"

        return PipelineResult(
            data=parsed,
            extraction_path=bundle.extraction_path,
            confidence=confidence,
            needs_review=needs_review,
            flags=bundle.flags,
            metadata={
                "pipeline_mode": "modern",
                "text_branch": bundle.branch,
                "token_usage": {"total_tokens": total_tokens},
                "estimated_cost": total_cost,
                "models": [settings.llm_deepseek_model],
                "steps": steps,
                **bundle.metadata,
            },
            text_extraction=text_extraction,
            extractions=[extraction],
        )
