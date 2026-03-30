import os
from typing import Any
import openai

from latam_eval.models.base import BaseModelAdapter


class UniversalOpenAIAdapter(BaseModelAdapter):
    """
    Universal Adapter for any LLM API that is compatible with the OpenAI SDK.
    This enables connecting to hundreds of free providers from the
    awesome-free-llm-apis list (e.g., Groq, Together AI, DeepInfra) effortlessly.
    """

    def __init__(
        self,
        model_name: str,
        base_url: str,
        api_key_env_var: str = "OPENAI_API_KEY",
        **kwargs: Any,
    ) -> None:
        """
        Initializes the UniversalOpenAIAdapter.

        Args:
            model_name (str): The specific model identifier (e.g., "llama3-8b-8192").
            base_url (str): The provider's OpenAI-compatible endpoint URL.
            api_key_env_var (str): Name of the env variable holding the API key.
            **kwargs: Extra parameters like temperature.
        """
        super().__init__(model_name=model_name, **kwargs)

        self.api_key = os.getenv(api_key_env_var)
        if not self.api_key:
            raise ValueError(
                f"{api_key_env_var} environment variable is not set. "
                "Please configure it to use this API provider."
            )

        self.client = openai.OpenAI(api_key=self.api_key, base_url=base_url)

    def generate_response(self, prompt: str, **kwargs: Any) -> str:
        """
        Generates a text response from the universal API.

        Args:
            prompt (str): The input text/instruction in Guarani/Jopara.
            **kwargs: Generation override parameters.

        Returns:
            str: the generated output.
        """
        gen_params = {**self.config, **kwargs}

        messages = [
            {
                "role": "system",
                "content": "You are a helpful AI assistant evaluating texts.",
            },
            {"role": "user", "content": prompt},
        ]

        try:
            response = self.client.chat.completions.create(
                model=self.model_name, messages=messages, **gen_params
            )
            return response.choices[0].message.content or ""
        except Exception as e:
            print(f"Error calling the universal API ({self.model_name}): {e}")
            raise
