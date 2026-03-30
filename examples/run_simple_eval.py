import os
import sys

# Ensure the root path is in PYTHONPATH
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from latam_eval.models.mock_adapter import MockAdapter
from latam_eval.datasets.loader import DatasetLoader
from latam_eval.models.gemini_adapter import GeminiAdapter
from latam_eval.models.universal_api_adapter import UniversalOpenAIAdapter


def main():
    print("=== Latam Evaluation Framework: Demo Run ===")

    # Load Sample Dataset
    dataset_path = os.path.join(
        os.path.dirname(__file__), "..", "data", "sample_guarani.json"
    )
    loader = DatasetLoader(dataset_path)
    dataset = loader.load()
    print(f"Loaded {len(dataset)} examples from the dataset.\n")

    # Initialize Mock Model
    # Since we don't have API keys active, we use MockAdapter to demonstrate the Multimodel Architecture
    model1 = MockAdapter(model_name="LatamGPT-Mock", temperature=0.7)
    model2 = MockAdapter(model_name="GPT-4o-Mock", temperature=0.0)
    gemini_model = GeminiAdapter(model_name="gemini-2.5-flash", temperature=0.7)
    
    models_to_evaluate = [model1, model2, gemini_model]
    # To use a free API from the Awesome List (e.g., Groq), simply export GROQ_API_KEY
    if os.getenv("GROQ_API_KEY"):
        models_to_evaluate.append(
            UniversalOpenAIAdapter(
                model_name="llama-3.3-70b-versatile",
                base_url="https://api.groq.com/openai/v1",
                api_key_env_var="GROQ_API_KEY",
                temperature=0.7
            )
        )
    else:
        print("Note: Export GROQ_API_KEY to see a free model from Groq in action!")
        
    # Another example: Together AI
    if os.getenv("TOGETHER_API_KEY"):
        models_to_evaluate.append(
            UniversalOpenAIAdapter(
                model_name="meta-llama/Llama-2-70b-chat-hf",
                base_url="https://api.together.xyz/v1",
                api_key_env_var="TOGETHER_API_KEY",
                temperature=0.7
            )
        )

    # Run Evaluation Loop
    for example in dataset:
        print(f"--- Evaluando Ejemplo ID: {example['id']} ---")
        print(f"Instrucción: {example['instruction']}")
        print(f"Respuesta Esperada: {example['expected_response']}\n")

        for model in models_to_evaluate:
            print(f"-> Modelo: {model.model_name}")
            response = model.generate_response(example["instruction"])
            print(f"Respuesta: {response}\n")

        print("-" * 40 + "\n")

    print("Demo Run Completed. Arquitectura Multimodelo Funcional y Probada.")


if __name__ == "__main__":
    main()
