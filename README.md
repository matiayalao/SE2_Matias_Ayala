# Latam-Eval-Framework

Framework de evaluación agnóstico y robusto para medir el desempeño de LLMs frente a
modelos fundacionales (GPT, Gemini, Claude, Llama, Gemma, etc.), con un enfoque especial
en lenguas indígenas americanas como el **Guaraní / Jopará** (Paraguay).

---

## Características

### Entregable 1 — Arquitectura base
- **Patrón Strategy/Adapter:** Integración sencilla de cualquier proveedor de LLMs.
- **Gestión de Dataset:** Carga estructurada de pares instrucción/respuesta en JSON y CSV.
- **Configuración Modular:** Parámetros definidos en archivos `YAML`.
- **Código Pythonico:** 100% `PEP8` y `docstrings` en todas las clases y funciones.

### Entregable 2 — CLI y métricas
- **CLI `eval-llm`:** Evaluaciones completas desde la línea de comandos.
- **Métricas NLP:** ROUGE-L, BLEU y METEOR para evaluar calidad de respuestas.

### Entregable 3 — Análisis comparativo (Cost-Performance Ratio)
- **Similitud de Embeddings:** Métrica semántica usando `mmaguero/multilingual-bert-gn-base-cased`
  (BERT entrenado específicamente en guaraní). Mide coherencia cultural más allá de n-gramas.
- **Cost-Performance Ratio:** Análisis de desempeño compuesto (`0.6×METEOR + 0.4×Emb.Sim`)
  frente al costo estimado (USD/1M tokens) y latencia de cada modelo.
- **Visualizaciones comparativas** generadas por `generate_report.py`:
  - `comparativa_metricas.png` — ROUGE-L, BLEU, METEOR y Emb. Similarity por modelo.
  - `cost_performance.png` — Desempeño compuesto vs. Costo.
  - `latency_performance.png` — Desempeño compuesto vs. Latencia.
- **Reporte final:** `reporte_final.md` con análisis técnico completo.

---

## Estructura del Proyecto

```
SE2/
├── latam_eval/
│   ├── cli.py                      # CLI: eval-llm
│   ├── evaluator.py                # Motor de evaluación + métricas
│   ├── datasets/
│   │   └── loader.py               # Carga JSON / CSV
│   ├── models/
│   │   ├── base.py                 # ABC BaseModelAdapter
│   │   ├── gemini_adapter.py       # Google Gemini
│   │   ├── openai_adapter.py       # Groq / OpenAI-compatible
│   │   ├── universal_api_adapter.py
│   │   └── mock_adapter.py         # Simulador local (sin API key)
│   └── utils/
│       └── config.py               # Carga YAML
├── data/
│   └── dataset_guarani_2.json      # Dataset de evaluación
├── tests/
│   ├── test_evaluator.py
│   └── test_cli.py
├── generate_report.py              # Genera las 3 visualizaciones PNG
├── report.json                     # Resultados: llama-3.1-8b-instant
├── report2.json                    # Resultados: mock
├── reporte_final.md                # Reporte técnico final (Entrega 3)
├── comparativa_metricas.png
├── cost_performance.png
├── latency_performance.png
├── config.yaml                     # Configuración de ejemplo
├── setup.py
└── requirements.txt
```

---

## Instalación y Configuración

### 1. Clonar y activar entorno virtual

```bash
git clone <url-del-repo>
cd SE2
python -m venv .venv
source .venv/bin/activate          # Linux/macOS
# .venv\Scripts\activate           # Windows
```

### 2. Instalar dependencias

```bash
pip install -r requirements.txt
pip install -e .                   # Instala el paquete y el CLI eval-llm
```

### 3. Configurar API Keys (solo para modelos reales)

```bash
export GROQ_API_KEY="gsk-..."       # Para llama, mixtral, etc. vía Groq (gratis)
export GEMINI_API_KEY="AIza-..."    # Para modelos Gemini
export OPENAI_API_KEY="sk-..."      # Para modelos GPT
```

> Si no tenés API keys, usá el modelo `mock` — funciona sin ninguna clave.

---

## Cómo probarlo (demo rápida sin API key)

### Demo instantánea con modelo mock

```bash
source .venv/bin/activate
eval-llm --models mock --dataset data/dataset_guarani_2.json --output mi_reporte.json
```

Eso corre la evaluación completa con el modelo simulado y guarda el resultado en `mi_reporte.json`.

### Generar las visualizaciones comparativas (Entrega 3)

```bash
python generate_report.py
```

Genera los 3 gráficos PNG en la raíz del proyecto.

### Correr los tests

```bash
python -m pytest tests/ -v
```

### Verificar PEP8

```bash
flake8 latam_eval/ generate_report.py
```

---

## Uso del CLI `eval-llm`

### Opción 1: Con archivo de configuración (recomendado)

```bash
eval-llm --config config.yaml
```

**Ejemplo de `config.yaml`:**
```yaml
models:
  - mock                      # Sin API key
  # - llama-3.1-8b-instant    # Requiere GROQ_API_KEY
  # - gemini-2.5-flash        # Requiere GEMINI_API_KEY
dataset: data/dataset_guarani_2.json
output: report.json
parameters:
  temperature: 0.7
  top_p: 0.9
```

### Opción 2: Por línea de comandos

```bash
eval-llm --models mock,llama-3.1-8b-instant \
         --dataset data/dataset_guarani_2.json \
         --output reporte_nuevo.json
```

### Parámetros disponibles

| Parámetro | Descripción |
|-----------|-------------|
| `--config` | Ruta al archivo YAML (si se usa, `--models` y `--dataset` son opcionales) |
| `--models` | Lista de modelos separados por coma |
| `--dataset` | Ruta al archivo JSON del dataset |
| `--output` | Ruta de salida del reporte (default: `results.json`) |

**Modelos soportados:**

| Nombre | Proveedor | API Key necesaria |
|--------|-----------|-------------------|
| `mock` | Local | ❌ Ninguna |
| `local-gemma` | LM Studio (puerto 1234) | ❌ Ninguna |
| `llama-3.1-8b-instant` | Groq | `GROQ_API_KEY` |
| `gemini-2.5-flash` | Google | `GEMINI_API_KEY` |
| `gpt-4o` / `gpt-4o-mini` | OpenAI | `OPENAI_API_KEY` |

### Estructura del dataset JSON

```json
[
  {
    "id": "q1",
    "category": "traduccion",
    "instruction": "Traduce 'Hola' al guaraní.",
    "expected_response": "Mba'éichapa"
  }
]
```

---

## Configuración de Modelo Local (LM Studio / Gemma)

Para evaluar sin depender de APIs externas:

1. Descargá [LM Studio](https://lmstudio.ai/) e instalá un modelo (ej. `gemma-2-9b-it`).
   ![Descarga de Gemma en LM Studio](captura1.png)
2. Configurá el modelo según tu hardware.
   ![Configuración de LM Studio](captura2.png)
3. Iniciá el servidor local en `http://127.0.0.1:1234`.
   ![Servidor LM Studio corriendo](captura3.png)
4. Usá el nombre `local-gemma` en el CLI:
   ```bash
   eval-llm --models local-gemma --dataset data/dataset_guarani_2.json --output report.json
   ```
