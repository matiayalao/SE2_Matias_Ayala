import os
from typing import Any
import openai

from latam_eval.models.base import BaseModelAdapter


class OpenAIAdapter(BaseModelAdapter):
    """
    Adapter for the OpenAI API (e.g., GPT-4o, GPT-3.5).
    """

    def __init__(self, model_name: str = "gpt-4o", **kwargs: Any) -> None:
        """
        Initializes the OpenAIAdapter.

        Args:
            model_name (str): Specifies the OpenAI model like "gpt-4o".
            **kwargs: Extra parameters like temperature, max_tokens, etc.
        """
        super().__init__(model_name=model_name, **kwargs)

        # Load the API Key from environment
        self.api_key = os.getenv("OPENAI_API_KEY")
        if not self.api_key:
            raise ValueError(
                "OPENAI_API_KEY environment variable is not set. "
                "Please configure it to use the OpenAI models."
            )

        self.client = openai.OpenAI(api_key=self.api_key)

    def generate_response(self, prompt: str, **kwargs: Any) -> str:
        """
        Generates a text response from the OpenAI API.

        Args:
            prompt (str): The input text/instruction in Guarani/Jopara.
            **kwargs: Generation override parameters.

        Returns:
            str: the generated output.
        """
        # Merge adapter-level config with local generation arguments
        gen_params = {**self.config, **kwargs}

        messages = [
            {"role": "system", "content": "You are a helpful AI assistant."},
            {"role": "user", "content": prompt},
        ]

        try:
            response = self.client.chat.completions.create(
                model=self.model_name, messages=messages, **gen_params
            )
            return response.choices[0].message.content or ""
        except Exception as e:
            # Note: in a production setting we might use a dedicated logger
            print(f"Error calling OpenAI API: {e}")
            raise
