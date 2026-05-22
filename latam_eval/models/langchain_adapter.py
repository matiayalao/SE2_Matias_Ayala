import os
from typing import Any

from langchain_core.messages import HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI
from langchain_anthropic import ChatAnthropic

from latam_eval.models.base import BaseModelAdapter


class LangchainAdapter(BaseModelAdapter):
    """
    Adapter for models using LangChain, allowing generic support
    for any model integrated via standard providers (Anthropic, OpenRouter, etc.).
    This is especially useful for fetching LatamGPT via HuggingFace Hub or OpenRouter
    and Claude via Anthropic natively.
    """

    def __init__(
        self, model_name: str, provider: str = "openrouter", **kwargs: Any
    ) -> None:
        """
        Initializes the LangchainAdapter for a versatile connection.

        Args:
            model_name (str): Full identifier, e.g. "anthropic/claude-3-opus"
            provider (str): Usually "openrouter", "anthropic", or "openai".
            **kwargs: Extra settings such as temperature.
        """
        super().__init__(model_name=model_name, **kwargs)
        self.provider = provider

        if self.provider == "anthropic":
            if not os.getenv("ANTHROPIC_API_KEY"):
                raise ValueError("ANTHROPIC_API_KEY environment variable is not set.")
            self.llm = ChatAnthropic(model=self.model_name, **self.config)

        elif self.provider == "openrouter":
            if not os.getenv("OPENROUTER_API_KEY"):
                raise ValueError("OPENROUTER_API_KEY environment variable is not set.")
            # OpenRouter acts like OpenAI interface
            self.llm = ChatOpenAI(
                base_url="https://openrouter.ai/api/v1",
                api_key=os.getenv("OPENROUTER_API_KEY"),
                model=self.model_name,
                **self.config,
            )

        elif self.provider == "local":
            # LM Studio or any local OpenAI compatible server
            self.llm = ChatOpenAI(
                base_url="http://localhost:1234/v1",
                api_key="lm-studio", # API key is required by the SDK but ignored by LM Studio
                model=self.model_name,
                **self.config,
            )

        else:
            raise ValueError(
                f"Provider {self.provider} not currently supported natively."
            )

    def generate_response(self, prompt: str, **kwargs: Any) -> str:
        """
        Generates a standard chat completion using the LangChain wrapper.

        Args:
            prompt (str): Input prompt for the chat model.

        Returns:
            str: Output string from the Language Model.
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

        messages = [
            SystemMessage(content=system_content),
            HumanMessage(content=prompt),
        ]

        # Bind any temporary generation kwargs specifically for this run
        bound_llm = self.llm.bind(**kwargs) if kwargs else self.llm

        try:
            response = bound_llm.invoke(messages)
            if isinstance(response.content, str):
                return response.content
            # in some edge cases with tools/complex returns, content might be a list
            return str(response.content)
        except Exception as e:
            print(f"Error calling {self.provider} API: {e}")
            raise
