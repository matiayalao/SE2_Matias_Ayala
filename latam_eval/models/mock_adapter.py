import time
from typing import Any

from latam_eval.models.base import BaseModelAdapter


class MockAdapter(BaseModelAdapter):
    """
    A dummy, mock adapter to be used for local testing and debugging
    of the framework without incurring usage costs for API Keys.
    """

    def __init__(self, model_name: str = "latamgpt-mock", **kwargs: Any) -> None:
        """
        Initializes the Mock Adapter.
        """
        super().__init__(model_name=model_name, **kwargs)

    def generate_response(self, prompt: str, **kwargs: Any) -> str:
        """
        Simulates generation by echoing the prompt with a predefined format
        after a brief delay resembling network latency.
        """
        # Inject standard latency
        time.sleep(0.5)

        # Simulated responses based on keywords could be added here
        # For simplicity, returning a formatted echoing text.
        return (
            f"[[ Mock response from {self.model_name} ]]\n"
            f"Recibí este prompt: '{prompt}'.\n"
            f"Mba'éichapa, esto es una prueba en Jopará/Guaraní simulada."
        )
