from dataclasses import dataclass

from app.config import settings
from app.services.llm.deepinfra import DeepInfraClient
from app.services.llm.json_extractor import JSONExtractor
from app.services.llm.prompt import build_messages, get_invoice_json_schema


@dataclass
class LLMCompletionResult:
    raw_output: str
    parsed_data: dict
    model: str
    prompt_tokens: int
    completion_tokens: int
    estimated_cost: float
    structured_output: bool


class InvoiceLLMExtractor:
    """Single-model invoice extraction using config-driven DeepInfra routing."""

    def __init__(self, api_key: str | None = None, model: str | None = None):
        api_key = api_key or settings.deepinfra_api_key
        if not api_key:
            raise ValueError("DEEPINFRA_API_KEY is not set")
        self.client = DeepInfraClient(api_key, model=model)
        self.model = self.client.model

    def extract(self, filename: str, source_text: str, text_source: str) -> LLMCompletionResult:
        messages = build_messages(filename, source_text)
        response = self._call_with_fallback(messages)
        content = response.choices[0].message.content or ""
        parsed = self._parse_json(content)
        if not parsed.get("original_filename"):
            parsed["original_filename"] = filename
        if not parsed.get("source"):
            parsed["source"] = text_source

        usage = response.usage
        return LLMCompletionResult(
            raw_output=content,
            parsed_data=parsed,
            model=self.model,
            prompt_tokens=usage.prompt_tokens,
            completion_tokens=usage.completion_tokens,
            estimated_cost=getattr(usage, "estimated_cost", 0.0),
            structured_output=settings.llm_use_structured_output,
        )

    def _call_with_fallback(self, messages: list[dict]):
        if not settings.llm_use_structured_output:
            return self.client.get_chat_completion(messages)

        try:
            return self.client.get_json_object_completion(messages)
        except Exception:
            try:
                return self.client.get_structured_completion(
                    messages,
                    get_invoice_json_schema(),
                )
            except Exception:
                return self.client.get_chat_completion(messages)

    def _parse_json(self, content: str) -> dict:
        try:
            return JSONExtractor.extract_json(content, sanitize=False)
        except ValueError as exc:
            raise ValueError(f"Failed to parse LLM JSON output: {exc}") from exc
