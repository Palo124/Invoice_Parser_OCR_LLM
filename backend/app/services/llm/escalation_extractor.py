from dataclasses import dataclass

from app.config import settings
from app.services.llm.deepinfra import DeepInfraClient
from app.services.llm.escalation_prompt import build_escalation_prompt
from app.services.llm.json_extractor import JSONExtractor
from app.services.llm.prompt import get_invoice_json_schema


@dataclass
class EscalationCompletionResult:
    raw_output: str
    parsed_data: dict
    model: str
    prompt_tokens: int
    completion_tokens: int
    estimated_cost: float


class InvoiceEscalationExtractor:
    """Stronger-model re-extraction for flagged invoices."""

    def __init__(self, api_key: str | None = None, model: str | None = None):
        api_key = api_key or settings.deepinfra_api_key
        if not api_key:
            raise ValueError("DEEPINFRA_API_KEY is not set")
        self.client = DeepInfraClient(api_key, model=model or settings.llm_escalation_model)
        self.model = self.client.model

    def extract(
        self,
        filename: str,
        source_text: str,
        current_json: dict,
        *,
        vision_json: dict | None = None,
        validation_errors: list[dict] | None = None,
        triggers: list[str] | None = None,
    ) -> EscalationCompletionResult:
        prompt = build_escalation_prompt(
            filename,
            source_text,
            current_json,
            vision_json=vision_json,
            validation_errors=validation_errors,
            triggers=triggers,
        )
        messages = [{"role": "user", "content": prompt}]
        response = self._call_with_fallback(messages)
        content = response.choices[0].message.content or ""
        parsed = self._parse_json(content)
        if not parsed.get("original_filename"):
            parsed["original_filename"] = filename
        if not parsed.get("source"):
            parsed["source"] = "escalation:llm"

        usage = response.usage
        return EscalationCompletionResult(
            raw_output=content,
            parsed_data=parsed,
            model=self.model,
            prompt_tokens=usage.prompt_tokens,
            completion_tokens=usage.completion_tokens,
            estimated_cost=getattr(usage, "estimated_cost", 0.0),
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
            raise ValueError(f"Failed to parse escalation model JSON output: {exc}") from exc
