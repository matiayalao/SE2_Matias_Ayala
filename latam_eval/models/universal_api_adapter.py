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

        # 1. Definir un contenido de sistema por defecto
        system_content = "You are a helpful AI assistant evaluating texts."

        # 2. Manejo dinámico de Temperature
        if "temperature" in gen_params:
            temp = gen_params["temperature"]
            if temp <= 0.3:
                system_content = (
                    "Eres un evaluador estricto. Responde ÚNICAMENTE con la traducción o la respuesta exacta solicitada. "
                    "No des explicaciones, ni contexto, ni opciones adicionales."
                )
            elif temp >= 0.7:
                system_content = (
                    "Eres un asistente conversacional. Provee la respuesta solicitada, pero también puedes incluir una breve explicación o ejemplos cuando sea pertinente."
                )

        # 3. Manejo dinámico de Top_P (solo si temperature no definió el prompt)
        elif "top_p" in gen_params:
            if gen_params["top_p"] < 0.5:
                system_content = (
                    "Responde de forma concisa y directa, sin elaboraciones innecesarias."
                )
            else:
                system_content = (
                    "Puedes ser más elaborado en tu respuesta, ofreciendo contexto adicional si lo consideras útil."
                )

        # 4. Preservar system_instruction explícito del usuario
        if "system_instruction" in gen_params:
            system_content = gen_params["system_instruction"]
            gen_params.pop("system_instruction", None)

        messages = [
            {"role": "system", "content": system_content},
            {"role": "user", "content": prompt},
        ]

        try:
            response = self.client.chat.completions.create(
                model=self.model_name,
                messages=messages,
                **gen_params,
            )
            return response.choices[0].message.content or ""
        except Exception as e:
            print(f"Error calling the universal API ({self.model_name}): {e}")
            raise
