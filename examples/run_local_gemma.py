import sys
import os

# Asegurarnos de que latam_eval está en el PYTHONPATH
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from latam_eval.models.langchain_adapter import LangchainAdapter

def main():
    print("Iniciando conexión con LM Studio (Local)...")
    
    # Configuramos el adaptador para usar nuestro nuevo proveedor local
    # El nombre del modelo puede ser cualquiera, ya que LM Studio usa el que está cargado.
    # Pero por buena práctica usamos el que vemos en la UI.
    adapter = LangchainAdapter(
        model_name="gemma-2-9b-it-simpo",
        provider="local",
        temperature=0.7
    )
    
    print("✓ Conexión lista.\n")
    
    prompt = "Mba'éichapa reiko? Traduce esto al español y explica brevemente."
    print(f"Usuario: {prompt}")
    print("-" * 50)
    
    # Hacemos la consulta
    print("Modelo (Gemma 2 9B Local):")
    response = adapter.generate_response(prompt)
    print(response)
    print("-" * 50)
    
if __name__ == "__main__":
    main()
