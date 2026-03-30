# Latam-Eval-Framework

Framework de evaluación agnóstico y robusto para medir el desempeño de LatamGPT frente a otros modelos fundacionales (ALIA, GPT, Gemini, Claude, Llama, Gemma, etc.), con un enfoque particular en lenguas indígenas americanas como el Guaraní.

## Características (Entregable 1)
- **Arquitectura Multimodelo:** Basada en el patrón `Strategy/Adapter` para la fácil integración de diversos proveedores (OpenAI, LangChain/OpenRouter, Mocking local).
- **Gestión de Dataset:** Carga estructurada de pares de instrucciones y respuestas.
- **Configuración Modular:** Configuración basada en archivos `YAML`.
- **Código Pythonico:** Implementación siguiendo `PEP8` y con cobertura total de `docstrings`.

## Estructura del Proyecto

```
latam-eval-framework/
│
├── latam_eval/
│   ├── models/                # Adaptadores de LLMs (Base, OpenAI, Langchain, Mock)
│   ├── datasets/              # Manejo y carga de datos JSON/JSONL
│   ├── utils/                 # Herramientas utilitarias (e.g. lector de config)
│
├── data/                      # Datasets de validación (Guaraní-Español)
├── examples/                  # Scripts de pruba rápidos
├── tests/                     # Tests unitarios 
```

## Requisitos y Configuración

1. Instalar dependencias:
   ```bash
   pip install -r requirements.txt
   ```
2. Configurar variables de entorno si se utilizarán las API reales (no mocks):
   ```bash
   export OPENAI_API_KEY="sk-..."
   export OPENROUTER_API_KEY="sk-or-v1-..."
   export ANTHROPIC_API_KEY="sk-ant-..."
   ```
3. Activar el entorno virtual
   ```bash
   source .venv/bin/activate
   ```

## Ejecutar una prueba base

Puedes ejecutar el ejemplo rápido que hace uso de los `MockAdapter` para probar el flujo sin consumir tokens reales:

```bash
python examples/run_simple_eval.py
```
