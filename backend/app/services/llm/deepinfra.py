from openai import OpenAI


class DeepInfraClient:
    def __init__(self, api_key: str, model: str):
        self.model = model
        self.client = OpenAI(
            api_key=api_key,
            base_url="https://api.deepinfra.com/v1/openai",
        )

    def get_chat_completion(self, messages: list[dict]):
        return self.client.chat.completions.create(
            model=self.model,
            messages=messages,
        )
