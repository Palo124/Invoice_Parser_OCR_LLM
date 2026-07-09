from typing import Any

from openai import OpenAI

from app.config import settings


class DeepInfraClient:
    def __init__(self, api_key: str, model: str | None = None):
        self.model = model or settings.llm_primary_model
        self.client = OpenAI(
            api_key=api_key,
            base_url="https://api.deepinfra.com/v1/openai",
        )

    def get_chat_completion(
        self,
        messages: list[dict],
        *,
        temperature: float | None = None,
        response_format: dict[str, Any] | None = None,
    ):
        cleaned = [
            {"role": message["role"], "content": message["content"]}
            for message in messages
        ]
        kwargs: dict[str, Any] = {
            "model": self.model,
            "messages": cleaned,
            "temperature": settings.llm_temperature if temperature is None else temperature,
        }
        if response_format is not None:
            kwargs["response_format"] = response_format

        return self.client.chat.completions.create(**kwargs)

    def get_json_object_completion(self, messages: list[dict], *, temperature: float | None = None):
        return self.get_chat_completion(
            messages,
            temperature=temperature,
            response_format={"type": "json_object"},
        )

    def get_structured_completion(
        self,
        messages: list[dict],
        schema: dict,
        *,
        schema_name: str = "invoice_data",
        temperature: float | None = None,
    ):
        return self.get_chat_completion(
            messages,
            temperature=temperature,
            response_format={
                "type": "json_schema",
                "json_schema": {
                    "name": schema_name,
                    "schema": schema,
                    "strict": False,
                },
            },
        )
