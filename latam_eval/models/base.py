from abc import ABC, abstractmethod
from typing import Any


class BaseModelAdapter(ABC):
    """
    Clase abstracta para los modelos, define el comportamiento que deben seguir los adaptadores
    """

    def __init__(self, model_name: str, **kwargs: Any) -> None:
        """
        Inicializa el adaptador con el nombre del modelo y parámetros adicionales
        """
        self.model_name = model_name
        self.config = kwargs

    @abstractmethod
    def generate_response(self, prompt: str, **kwargs: Any) -> str:
        """
        Genera una respuesta del modelo dado un prompt específico.
        """
        pass
