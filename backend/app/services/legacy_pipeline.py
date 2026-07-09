from pathlib import Path

import numpy as np
from PIL import Image

from app.config import settings
from app.services.exceptions import ProcessingCancelled
from app.services.llm.extractor import InvoiceLLMExtractor
from app.services.merge.tmr import merge_extractions_with_flags
from app.services.ocr.paddle_ocr_processor import PaddleOCRProcessor
from app.services.ocr.pytesseract_ocr_processor import PytesseractOCRProcessor
from app.services.pipeline_common import finalize_pipeline_result
from app.services.preprocessing.deskew import ImageDeskewer
from app.services.preprocessing.pdf_converter import PDFToImageConverter
from app.services.text_extraction.ocr_comparator import compare_ocr_texts
from app.services.types import (
    CancelCheck,
    ExtractionResult,
    PipelineResult,
    ProgressCallback,
    TextExtractionResult,
)
from app.services.validation import InvoiceValidationService


class LegacyPipeline:
    """Legacy 2x OCR + 2x LLM + field-level merge pipeline (kept behind LEGACY_PIPELINE)."""

    def __init__(self):
        if not settings.deepinfra_api_key:
            raise ValueError("DEEPINFRA_API_KEY is not set")

        self.ocr_paddle = PaddleOCRProcessor(
            lang=settings.ocr_paddle_lang,
            use_gpu=settings.ocr_paddle_gpu,
        )
        self.ocr_tesseract = PytesseractOCRProcessor(
            tesseract_cmd=settings.tesseract_cmd,
            lang=settings.ocr_tesseract_lang,
        )
        self.llm_deepseek = InvoiceLLMExtractor(model=settings.llm_deepseek_model)
        self.llm_llama = InvoiceLLMExtractor(model=settings.llm_llama_model)
        self.validator = InvoiceValidationService()

    def _load_images(self, file_path: Path) -> tuple[list[np.ndarray], str]:
        ext = file_path.suffix.lower()
        if ext == ".pdf":
            converter = PDFToImageConverter(poppler_path=None)
            pages = converter.convert_pdf_to_images(str(file_path), dpi=settings.ocr_pdf_dpi)
            return [np.array(page.convert("RGB")) for page in pages], "pdf"
        if ext in {".png", ".jpg", ".jpeg"}:
            with Image.open(file_path) as img:
                return [np.array(img.convert("RGB"))], "image"
        raise ValueError("Unsupported file format. Use PDF, PNG, or JPEG.")

    def _deskew_pages(self, images: list[np.ndarray]) -> list[np.ndarray]:
        deskewed: list[np.ndarray] = []
        for image in images:
            try:
                deskewer = ImageDeskewer(image)
                rotated_image, _ = deskewer.deskew()
            except ValueError:
                rotated_image = image
            deskewed.append(rotated_image)
        return deskewed

    def _run_ocr_engine(self, pages: list[np.ndarray], extract_fn, threshold: int) -> str:
        page_texts: list[str] = []
        for page in pages:
            pil_image = Image.fromarray(page)
            page_texts.append(extract_fn(pil_image, threshold))
        return "\n".join(page_texts)

    def _ocr_step(self, engine: str, text: str) -> dict:
        return {
            "engine": engine,
            "text": text,
            "char_count": len(text),
            "preview": text[:500] + ("…" if len(text) > 500 else ""),
        }

    def process_file(
        self,
        file_path: Path,
        on_progress: ProgressCallback | None = None,
        should_cancel: CancelCheck | None = None,
    ) -> PipelineResult:
        filename = file_path.name
        steps: dict = {"ocr": [], "llm": []}

        def check_cancelled() -> None:
            if should_cancel and should_cancel():
                raise ProcessingCancelled()

        def report(stage: str) -> None:
            check_cancelled()
            if on_progress:
                on_progress(stage, steps)

        check_cancelled()
        images, source_type = self._load_images(file_path)
        steps["preprocessing"] = {
            "source_type": source_type,
            "page_count": len(images),
            "deskew": True,
        }
        report("preprocessing")

        deskewed_pages = self._deskew_pages(images)

        text_tesseract = self._run_ocr_engine(
            deskewed_pages,
            self.ocr_tesseract.extract_text_layout_from_pil,
            settings.ocr_tesseract_threshold,
        )
        steps["ocr"].append(self._ocr_step("tesseract", text_tesseract))
        report("ocr:tesseract")

        text_paddle = self._run_ocr_engine(
            deskewed_pages,
            self.ocr_paddle.extract_text_layout_from_pil,
            settings.ocr_paddle_threshold,
        )
        steps["ocr"].append(self._ocr_step("paddleocr", text_paddle))
        report("ocr:paddleocr")

        ocr_comparison = compare_ocr_texts(
            text_tesseract,
            text_paddle,
            agreement_threshold=settings.ocr_agreement_threshold,
        )
        steps["ocr_comparison"] = {
            "similarity": ocr_comparison.similarity,
            "agreement": ocr_comparison.agreement,
        }

        text_extraction = TextExtractionResult(
            text="\n\n--- TESSERACT ---\n\n".join([text_tesseract, text_paddle]),
            source="legacy:tesseract+paddle",
            confidence=1.0,
        )

        llm_jobs = [
            (self.llm_deepseek, "tesseract", text_tesseract, "llm:deepseek"),
            (self.llm_llama, "paddleocr", text_paddle, "llm:llama"),
        ]

        extractions: list[ExtractionResult] = []
        total_tokens = 0
        total_cost = 0.0

        for llm_extractor, ocr_engine, ocr_text, progress_stage in llm_jobs:
            check_cancelled()
            llm_result = llm_extractor.extract(filename, ocr_text, ocr_engine)
            total_tokens += llm_result.prompt_tokens + llm_result.completion_tokens
            total_cost += llm_result.estimated_cost

            extractions.append(
                ExtractionResult(
                    data=llm_result.parsed_data,
                    model=llm_result.model,
                    raw_output=llm_result.raw_output,
                    ocr_engine=ocr_engine,
                    prompt_tokens=llm_result.prompt_tokens,
                    completion_tokens=llm_result.completion_tokens,
                )
            )
            steps["llm"].append(
                {
                    "model": llm_result.model,
                    "ocr_engine": ocr_engine,
                    "raw_output": llm_result.raw_output,
                    "parsed_json": llm_result.parsed_data,
                    "prompt_tokens": llm_result.prompt_tokens,
                    "completion_tokens": llm_result.completion_tokens,
                    "structured_output": llm_result.structured_output,
                }
            )
            report(progress_stage)

        merge_result = merge_extractions_with_flags(
            extractions[0].data,
            extractions[1].data,
        )
        steps["tmr"] = {
            "merged_json": merge_result.merged,
            "disagreements": [error.to_dict() for error in merge_result.disagreements],
        }
        report("tmr")

        check_cancelled()
        report("validation")
        validation = self.validator.validate(
            merge_result.merged,
            raw_text=text_extraction.text,
            ocr_comparison=ocr_comparison,
            merge_disagreements=merge_result.disagreements,
        )

        return finalize_pipeline_result(
            data=merge_result.merged,
            extraction_path=settings.legacy_extraction_path,
            validation=validation,
            steps=steps,
            metadata={
                "pipeline_mode": "legacy",
                "token_usage": {"total_tokens": total_tokens},
                "estimated_cost": total_cost,
                "models": [item.model for item in extractions],
            },
            text_extraction=text_extraction,
            extractions=extractions,
            raw_text=text_extraction.text,
            llm_raw_json=extractions[0].raw_output,
            model_used=",".join(item.model for item in extractions),
        )
