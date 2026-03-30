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
        messages = [
            SystemMessage(content="You are a helpful AI assistant evaluating texts."),
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
