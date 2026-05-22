import os
from typing import Any
import openai

from latam_eval.models.base import BaseModelAdapter


class GroqAdapter(BaseModelAdapter):  # Le cambiamos el nombre para que sea coherente
    """
    Adapter for the Groq API (e.g., Llama 3, Gemma, Mixtral) using OpenAI SDK.
    """

# ... (el resto de tus imports se queda igual)

# Volvemos a llamarla OpenAIAdapter para que cli.py no rompa al importar
class OpenAIAdapter(BaseModelAdapter): 
    """
    Adapter for the Groq API using OpenAI SDK structure.
    """

    def __init__(self, model_name: str = "llama-3.1-8b-instant", **kwargs: Any) -> None:
        super().__init__(model_name=model_name, **kwargs)

        # Mantenemos tu lógica de Groq intacta aquí abajo
        self.api_key = os.getenv("GROQ_API_KEY")
        if not self.api_key:
            raise ValueError("GROQ_API_KEY environment variable is not set.")

        self.client = openai.OpenAI(
            api_key=self.api_key,
            base_url="https://api.groq.com/openai/v1"
        )
# ... (el resto del método generate_response se queda igual)

    def generate_response(self, prompt: str, **kwargs: Any) -> str:
        """
        Generates a text response from the Groq API (via OpenAI SDK),
        dynamically adjusting the system prompt based on temperature/top_p.
        """
        # Combinar la configuración base con los argumentos locales de generación
        gen_params = {**self.config, **kwargs}

        # 1. Definir un contenido de sistema por defecto
        system_content = "You are a helpful AI assistant."

        # 2. Manejo dinámico de Temperature (Misma lógica que usaste en Gemini)
        if "temperature" in gen_params:
            temp = gen_params["temperature"]
            if temp <= 0.3:
                print("Entró en 0.3\n")
                system_content = (
                    "Eres un evaluador estricto. Responde ÚNICAMENTE con la traducción o la respuesta exacta solicitada. "
                    "No des explicaciones, ni contexto, ni opciones adicionales."
                )
            elif temp >= 0.7:
                print("Entró en 0.7\n")
                system_content = (
                    "Eres un asistente conversacional. Provee la respuesta solicitada, pero también puedes incluir una breve explicación o ejemplos cuando sea pertinente."
                )
        
        # 3. Manejo dinámico de Top_P (Si la temperatura no definió el prompt antes)
        elif "top_p" in gen_params:
            if gen_params["top_p"] < 0.5:
                system_content = (
                    "Responde de forma concisa y directa, sin elaboraciones innecesarias."
                )
            else:
                system_content = (
                    "Puedes ser más elaborado en tu respuesta, ofreciendo contexto adicional si lo consideras útil."
                )

        # 4. Preservar system_instruction si viene explícitamente en el archivo de configuración
        if "system_instruction" in gen_params:
            print("Entró en system_instruction\n")
            system_content = gen_params["system_instruction"]
            # IMPORTANTE: Eliminamos 'system_instruction' del diccionario gen_params
            # porque la API de OpenAI/Groq no acepta ese parámetro ahí y lanzaría un error.
            gen_params.pop("system_instruction", None)
        
        print(f"\nSYSTEM PROMPT: {system_content}\n")

        # 5. Construir los mensajes inyectando el system_content dinámico
        messages = [
            {"role": "system", "content": system_content},
            {"role": "user", "content": prompt},
        ]

        try:
            # Enviamos los parámetros limpios (temperature, top_p, etc.) a la API
            response = self.client.chat.completions.create(
                model=self.model_name, 
                messages=messages, 
                **gen_params
            )
            return response.choices[0].message.content or ""
        except Exception as e:
            print(f"Error calling Groq API: {e}")
            raise