import json
from dataclasses import dataclass
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


@dataclass
class PipelineResult:
    data: dict
    token_usage: dict
    estimated_cost: float


class InvoicePipeline:
    def __init__(self):
        if not settings.deepinfra_api_key:
            raise ValueError("DEEPINFRA_API_KEY is not set")

        self.ocr_easy = EasyOCRProcessor(languages=["cs"], gpu=True)
        self.ocr_paddle = PaddleOCRProcessor(lang="cs", use_gpu=True)
        self.ocr_tesseract = PytesseractOCRProcessor(
            tesseract_cmd=settings.tesseract_cmd,
            lang="ces",
        )
        self.llm_deepseek = DeepInfraClient(
            settings.deepinfra_api_key,
            "deepseek-ai/DeepSeek-R1",
        )
        self.llm_llama = DeepInfraClient(
            settings.deepinfra_api_key,
            "meta-llama/Llama-3.3-70B-Instruct-Turbo",
        )
        self.llm_maverick = DeepInfraClient(
            settings.deepinfra_api_key,
            "meta-llama/Llama-4-Maverick-17B-128E-Instruct-FP8",
        )

    def _load_images(self, file_path: Path) -> list[np.ndarray]:
        ext = file_path.suffix.lower()
        if ext == ".pdf":
            converter = PDFToImageConverter(poppler_path=None)
            pages = converter.convert_pdf_to_images(str(file_path), dpi=300)
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
                self.ocr_tesseract.extract_text_layout_from_pil(pil_image, threshold=15)
            )
            texts_paddle.append(
                self.ocr_paddle.extract_text_layout_from_pil(pil_image, threshold=15)
            )
            texts_easy.append(
                self.ocr_easy.image_to_text_layout(pil_image, threshold=30)
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

        response_deepseek = self._call_llm(self.llm_deepseek, filename, text_tesseract)
        response_llama = self._call_llm(self.llm_llama, filename, text_paddle)
        response_maverick = self._call_llm(self.llm_maverick, filename, text_easy)

        parsed = []
        for response in (response_deepseek, response_llama, response_maverick):
            content = response.choices[0].message.content
            parsed.append(JSONExtractor.extract_json(content))

        merged = triple_modular_redundancy(parsed[0], parsed[1], parsed[2])

        total_tokens = sum(
            response.usage.prompt_tokens + response.usage.completion_tokens
            for response in (response_deepseek, response_llama, response_maverick)
        )
        total_cost = sum(
            getattr(response.usage, "estimated_cost", 0.0)
            for response in (response_deepseek, response_llama, response_maverick)
        )

        return PipelineResult(
            data=merged,
            token_usage={"total_tokens": total_tokens},
            estimated_cost=total_cost,
        )
