import os
from typing import Any

import openai

from latam_eval.models.base import BaseModelAdapter


class OpenAIAdapter(BaseModelAdapter):
    """
    Adapter for the Groq API (e.g., Llama 3, Gemma, Mixtral) using the
    OpenAI-compatible SDK interface.

    Dynamically adjusts the system prompt based on ``temperature`` and
    ``top_p`` generation parameters.
    """

    def __init__(self, model_name: str = "llama-3.1-8b-instant", **kwargs: Any) -> None:
        """
        Initializes the OpenAIAdapter pointing to the Groq endpoint.

        Args:
            model_name (str): The Groq model identifier to use.
            **kwargs: Extra generation parameters (e.g. temperature, top_p).
        """
        super().__init__(model_name=model_name, **kwargs)

        self.api_key = os.getenv("GROQ_API_KEY")
        if not self.api_key:
            raise ValueError("GROQ_API_KEY environment variable is not set.")

        self.client = openai.OpenAI(
            api_key=self.api_key, base_url="https://api.groq.com/openai/v1"
        )

    def generate_response(self, prompt: str, **kwargs: Any) -> str:
        """
        Generates a text response from the Groq API via the OpenAI SDK,
        dynamically adjusting the system prompt based on temperature/top_p.

        Args:
            prompt (str): The input text/instruction in Guarani/Jopara.
            **kwargs: Generation override parameters.

        Returns:
            str: The model-generated output text.
        """
        gen_params = {**self.config, **kwargs}

        # Default system content
        system_content = "You are a helpful AI assistant."

        # Dynamic Temperature handling
        if "temperature" in gen_params:
            temp = gen_params["temperature"]
            if temp <= 0.3:
                system_content = (
                    "Eres un evaluador estricto. Responde ÚNICAMENTE con la "
                    "traducción o la respuesta exacta solicitada. "
                    "No des explicaciones, ni contexto, ni opciones adicionales."
                )
            elif temp >= 0.7:
                system_content = (
                    "Eres un asistente conversacional. Provee la respuesta solicitada, "
                    "pero también puedes incluir una breve explicación o ejemplos "
                    "cuando sea pertinente."
                )

        # Dynamic Top_P handling (only if temperature did not set the prompt)
        elif "top_p" in gen_params:
            if gen_params["top_p"] < 0.5:
                system_content = (
                    "Responde de forma concisa y directa, sin elaboraciones innecesarias."
                )
            else:
                system_content = (
                    "Puedes ser más elaborado en tu respuesta, ofreciendo contexto "
                    "adicional si lo consideras útil."
                )

        # Preserve explicit system_instruction from the config
        if "system_instruction" in gen_params:
            system_content = gen_params.pop("system_instruction")

        messages = [
            {"role": "system", "content": system_content},
            {"role": "user", "content": prompt},
        ]

        try:
            response = self.client.chat.completions.create(
                model=self.model_name, messages=messages, **gen_params
            )
            return response.choices[0].message.content or ""
        except Exception as e:
            print(f"Error calling Groq API: {e}")
            raise
