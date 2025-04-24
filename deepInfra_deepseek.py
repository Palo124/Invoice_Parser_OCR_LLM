# deepInfra_deepseek.py

from openai import OpenAI

class DeepSeekInterface:
    """
    A class to interact with the DeepInfra OpenAI endpoint.
    """

    def __init__(self, api_key: str):
        """
        Initialize the DeepInfraOpenAI client with a specified API key and base URL.
        
        :param api_key: Your DeepInfra API key.
        :param base_url: The DeepInfra endpoint URL (e.g., "https://api.deepinfra.com/v1/openai").
        """
        self.openai = OpenAI(
            api_key=api_key,
            base_url="https://api.deepinfra.com/v1/openai"
        )

    def get_chat_completion(self, prompt):
        """
        Sends a chat completion request to the DeepInfra OpenAI endpoint.
        
        :param messages: A list of message dicts, e.g. [{"role": "user", "content": "Hello"}].
        :param model: Name of the model to use. Defaults to "deepseek-ai/DeepSeek-R1".
        :return: The full response object from openai.chat.completions.create().
        """
        response = self.openai.chat.completions.create(
            model="deepseek-ai/DeepSeek-R1",
            #model="meta-llama/Llama-4-Maverick-17B-128E-Instruct-FP8", #other llama
            messages=prompt
        )
        return response
