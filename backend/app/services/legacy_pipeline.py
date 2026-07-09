from pathlib import Path

import numpy as np
from PIL import Image

from app.config import settings
from app.services.exceptions import ProcessingCancelled
from app.services.llm.deepinfra import DeepInfraClient
from app.services.llm.json_extractor import JSONExtractor
from app.services.llm.prompt import get_prompt
from app.services.merge.tmr import triple_modular_redundancy
from app.services.ocr.paddle_ocr_processor import PaddleOCRProcessor
from app.services.ocr.pytesseract_ocr_processor import PytesseractOCRProcessor
from app.services.preprocessing.deskew import ImageDeskewer
from app.services.preprocessing.pdf_converter import PDFToImageConverter
from app.services.types import (
    CancelCheck,
    ExtractionResult,
    PipelineResult,
    ProgressCallback,
    TextExtractionResult,
)


class LegacyPipeline:
    """Legacy 2x OCR + 2x LLM + TMR pipeline (kept behind LEGACY_PIPELINE)."""

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
        self.llm_deepseek = DeepInfraClient(
            settings.deepinfra_api_key,
            settings.llm_deepseek_model,
        )
        self.llm_llama = DeepInfraClient(
            settings.deepinfra_api_key,
            settings.llm_llama_model,
        )

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

    def _call_llm(self, client: DeepInfraClient, filename: str, ocr_text: str):
        prompt = [{"role": "user", "content": get_prompt(filename, ocr_text)}]
        return client.get_chat_completion(prompt, temperature=0.0)

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

        text_extraction = TextExtractionResult(
            text="\n\n--- TESSERACT ---\n\n".join([text_tesseract, text_paddle]),
            source="legacy:tesseract+paddle",
            confidence=1.0,
        )

        llm_clients = [
            (self.llm_deepseek, settings.llm_deepseek_model, "tesseract", text_tesseract, "llm:deepseek"),
            (self.llm_llama, settings.llm_llama_model, "paddleocr", text_paddle, "llm:llama"),
        ]

        responses = []
        extractions: list[ExtractionResult] = []

        for client, model_name, ocr_engine, ocr_text, progress_stage in llm_clients:
            check_cancelled()
            response = self._call_llm(client, filename, ocr_text)
            responses.append(response)
            content = response.choices[0].message.content or ""
            parsed = JSONExtractor.extract_json(content)
            prompt_tokens = response.usage.prompt_tokens
            completion_tokens = response.usage.completion_tokens

            extractions.append(
                ExtractionResult(
                    data=parsed,
                    model=model_name,
                    confidence=settings.default_confidence,
                    raw_output=content,
                    ocr_engine=ocr_engine,
                    prompt_tokens=prompt_tokens,
                    completion_tokens=completion_tokens,
                )
            )
            steps["llm"].append(
                {
                    "model": model_name,
                    "ocr_engine": ocr_engine,
                    "raw_output": content,
                    "parsed_json": parsed,
                    "prompt_tokens": prompt_tokens,
                    "completion_tokens": completion_tokens,
                }
            )
            report(progress_stage)

        merged = triple_modular_redundancy(
            extractions[0].data,
            extractions[1].data,
            extractions[1].data,
        )
        steps["tmr"] = {"merged_json": merged}
        report("tmr")

        total_tokens = sum(
            response.usage.prompt_tokens + response.usage.completion_tokens
            for response in responses
        )
        total_cost = sum(
            getattr(response.usage, "estimated_cost", 0.0) for response in responses
        )

        return PipelineResult(
            data=merged,
            extraction_path=settings.legacy_extraction_path,
            confidence=settings.default_confidence,
            needs_review=False,
            flags=[],
            metadata={
                "pipeline_mode": "legacy",
                "token_usage": {"total_tokens": total_tokens},
                "estimated_cost": total_cost,
                "models": [item.model for item in extractions],
                "steps": steps,
            },
            text_extraction=text_extraction,
            extractions=extractions,
        )
