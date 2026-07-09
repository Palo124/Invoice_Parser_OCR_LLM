import base64
from dataclasses import dataclass
from io import BytesIO

import numpy as np
from PIL import Image

from app.config import settings
from app.services.llm.deepinfra import DeepInfraClient
from app.services.llm.json_extractor import JSONExtractor
from app.services.llm.vision_prompt import get_vision_prompt


@dataclass
class VisionCompletionResult:
    raw_output: str
    parsed_data: dict
    model: str
    prompt_tokens: int
    completion_tokens: int
    estimated_cost: float
    page_count: int


def _image_to_data_url(page: np.ndarray) -> str:
    pil_image = Image.fromarray(page)
    buffer = BytesIO()
    pil_image.save(buffer, format="JPEG", quality=85)
    encoded = base64.b64encode(buffer.getvalue()).decode("ascii")
    return f"data:image/jpeg;base64,{encoded}"


def _build_vision_messages(filename: str, page_images: list[np.ndarray]) -> list[dict]:
    content: list[dict] = [
        {"type": "text", "text": get_vision_prompt(filename, len(page_images))},
    ]
    for page in page_images:
        content.append(
            {
                "type": "image_url",
                "image_url": {"url": _image_to_data_url(page)},
            }
        )
    return [{"role": "user", "content": content}]


class InvoiceVisionExtractor:
    """Vision-model invoice extraction for complex scanned layouts."""

    def __init__(self, api_key: str | None = None, model: str | None = None):
        api_key = api_key or settings.deepinfra_api_key
        if not api_key:
            raise ValueError("DEEPINFRA_API_KEY is not set")
        self.client = DeepInfraClient(api_key, model=model or settings.llm_vision_model)
        self.model = self.client.model

    def extract(self, filename: str, page_images: list[np.ndarray]) -> VisionCompletionResult:
        if not page_images:
            raise ValueError("At least one page image is required for vision extraction")

        limited_pages = page_images[: settings.vision_max_pages]
        messages = _build_vision_messages(filename, limited_pages)
        response = self.client.get_vision_completion(messages)
        content = response.choices[0].message.content or ""
        parsed = self._parse_json(content)
        if not parsed.get("original_filename"):
            parsed["original_filename"] = filename
        if not parsed.get("source"):
            parsed["source"] = "vision:qwen3-vl"

        usage = response.usage
        return VisionCompletionResult(
            raw_output=content,
            parsed_data=parsed,
            model=self.model,
            prompt_tokens=usage.prompt_tokens,
            completion_tokens=usage.completion_tokens,
            estimated_cost=getattr(usage, "estimated_cost", 0.0),
            page_count=len(limited_pages),
        )

    def _parse_json(self, content: str) -> dict:
        try:
            return JSONExtractor.extract_json(content, sanitize=False)
        except ValueError as exc:
            raise ValueError(f"Failed to parse vision model JSON output: {exc}") from exc
