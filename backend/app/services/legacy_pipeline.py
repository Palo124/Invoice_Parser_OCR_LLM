from pathlib import Path

import numpy as np
from PIL import Image

from app.config import settings
from app.services.llm.deepinfra import DeepInfraClient
from app.services.llm.json_extractor import JSONExtractor
from app.services.llm.prompt import get_prompt
from app.services.merge.tmr import triple_modular_redundancy
from app.services.ocr.easyocr_processor import EasyOCRProcessor
from app.services.ocr.paddle_ocr_processor import PaddleOCRProcessor
from app.services.ocr.pytesseract_ocr_processor import PytesseractOCRProcessor
from app.services.preprocessing.deskew import ImageDeskewer
from app.services.preprocessing.pdf_converter import PDFToImageConverter
from app.services.types import ExtractionResult, PipelineResult, TextExtractionResult


class LegacyPipeline:
    """Original 3x OCR + 3x LLM + TMR pipeline."""

    def __init__(self):
        if not settings.deepinfra_api_key:
            raise ValueError("DEEPINFRA_API_KEY is not set")

        self.ocr_easy = EasyOCRProcessor(
            languages=[settings.ocr_easyocr_lang],
            gpu=settings.ocr_easyocr_gpu,
        )
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
        self.llm_maverick = DeepInfraClient(
            settings.deepinfra_api_key,
            settings.llm_maverick_model,
        )

    def _load_images(self, file_path: Path) -> list[np.ndarray]:
        ext = file_path.suffix.lower()
        if ext == ".pdf":
            converter = PDFToImageConverter(poppler_path=None)
            pages = converter.convert_pdf_to_images(str(file_path), dpi=settings.ocr_pdf_dpi)
            return [np.array(page) for page in pages]
        if ext in {".png", ".jpg", ".jpeg"}:
            return [np.array(Image.open(file_path))]
        raise ValueError("Unsupported file format. Use PDF, PNG, or JPEG.")

    def _ocr_pages(self, images: list[np.ndarray]) -> tuple[str, str, str]:
        texts_tesseract: list[str] = []
        texts_paddle: list[str] = []
        texts_easy: list[str] = []

        for image in images:
            try:
                deskewer = ImageDeskewer(image)
                rotated_image, _ = deskewer.deskew()
            except ValueError:
                rotated_image = image

            pil_image = Image.fromarray(rotated_image)
            texts_tesseract.append(
                self.ocr_tesseract.extract_text_layout_from_pil(
                    pil_image,
                    threshold=settings.ocr_tesseract_threshold,
                )
            )
            texts_paddle.append(
                self.ocr_paddle.extract_text_layout_from_pil(
                    pil_image,
                    threshold=settings.ocr_paddle_threshold,
                )
            )
            texts_easy.append(
                self.ocr_easy.image_to_text_layout(
                    pil_image,
                    threshold=settings.ocr_easyocr_threshold,
                )
            )

        return (
            "\n".join(texts_tesseract),
            "\n".join(texts_paddle),
            "\n".join(texts_easy),
        )

    def _call_llm(self, client: DeepInfraClient, filename: str, ocr_text: str):
        prompt = [{"role": "user", "content": get_prompt(filename, ocr_text)}]
        return client.get_chat_completion(prompt)

    def process_file(self, file_path: Path) -> PipelineResult:
        filename = file_path.name
        images = self._load_images(file_path)
        text_tesseract, text_paddle, text_easy = self._ocr_pages(images)

        combined_text = "\n\n--- TESSERACT ---\n\n".join(
            [text_tesseract, text_paddle, text_easy]
        )
        text_extraction = TextExtractionResult(
            text=combined_text,
            source="legacy:tesseract+paddle+easyocr",
            confidence=1.0,
        )

        llm_clients = [
            (self.llm_deepseek, settings.llm_deepseek_model, text_tesseract),
            (self.llm_llama, settings.llm_llama_model, text_paddle),
            (self.llm_maverick, settings.llm_maverick_model, text_easy),
        ]

        responses = []
        extractions: list[ExtractionResult] = []
        for client, model_name, ocr_text in llm_clients:
            response = self._call_llm(client, filename, ocr_text)
            responses.append(response)
            content = response.choices[0].message.content
            parsed = JSONExtractor.extract_json(content)
            extractions.append(
                ExtractionResult(
                    data=parsed,
                    model=model_name,
                    confidence=settings.default_confidence,
                )
            )

        merged = triple_modular_redundancy(
            extractions[0].data,
            extractions[1].data,
            extractions[2].data,
        )

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
            },
            text_extraction=text_extraction,
            extractions=extractions,
        )
