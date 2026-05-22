import os
from typing import Any
from google import genai
from google.genai import types

from latam_eval.models.base import BaseModelAdapter


class GeminiAdapter(BaseModelAdapter):
    """
    Adapter for the Google Gemini API using the new google-genai SDK.
    """

    def __init__(self, model_name: str = "gemini-2.5-flash", **kwargs: Any) -> None:
        """
        Initializes the GeminiAdapter.

        Args:
            model_name (str): The Gemini model to use.
            **kwargs: Extra parameters like temperature.
        """
        super().__init__(model_name=model_name, **kwargs)

        # Load the API Key from environment
        self.api_key = os.getenv("GEMINI_API_KEY")
        if not self.api_key:
            raise ValueError(
                "GEMINI_API_KEY environment variable is not set. "
                "Please configure it to use the Gemini models."
            )

        self.client = genai.Client(api_key=self.api_key)

    def generate_response(self, prompt: str, **kwargs: Any) -> str:
        """
        Generates a text response from the Gemini API.

        Args:
            prompt (str): The input text/instruction in Guarani/Jopara.
            **kwargs: Generation override parameters.

        Returns:
            str: the generated output.
        """
        gen_params = {**self.config, **kwargs}

        config_kwargs = {}
        # Temperature & top_p handling
        if "temperature" in gen_params:
            config_kwargs["temperature"] = gen_params["temperature"]
            # Choose a system instruction based on temperature thresholds
            temp = gen_params["temperature"]
            if temp <= 0.3:
                # Very conservative / strict response: only the exact answer
                config_kwargs["system_instruction"] = (
                    "Eres un evaluador estricto. Responde ÚNICAMENTE con la traducción o la respuesta exacta solicitada. "
                    "No des explicaciones, ni contexto, ni opciones adicionales."
                )
            elif temp >= 0.7:
                # More creative / verbose response
                config_kwargs["system_instruction"] = (
                    "Eres un asistente conversacional. Provee la respuesta solicitada, pero también puedes incluir una breve explicación o ejemplos cuando sea pertinente."
                )
        if "top_p" in gen_params:
            config_kwargs["top_p"] = gen_params["top_p"]
            # Optionally tweak system instruction when top_p is low/high
            if "system_instruction" not in config_kwargs:
                if gen_params["top_p"] < 0.5:
                    config_kwargs["system_instruction"] = (
                        "Responde de forma concisa y directa, sin elaboraciones innecesarias."
                    )
                else:
                    config_kwargs["system_instruction"] = (
                        "Puedes ser más elaborado en tu respuesta, ofreciendo contexto adicional si lo consideras útil."
                    )
        if "top_k" in gen_params:
            config_kwargs["top_k"] = gen_params["top_k"]
        if "max_tokens" in gen_params:
            config_kwargs["max_output_tokens"] = gen_params["max_tokens"]
        # Preserve explicit system_instruction if user supplied it directly
        if "system_instruction" in gen_params and "system_instruction" not in config_kwargs:
            config_kwargs["system_instruction"] = gen_params["system_instruction"]

        config = types.GenerateContentConfig(**config_kwargs) if config_kwargs else None

        try:
            response = self.client.models.generate_content(
                model=self.model_name,
                contents=prompt,
                config=config,
            )
            return response.text
        except Exception as e:
            print(f"Error calling Gemini API: {e}")
            raise
