"""
generate_report.py
==================
Script para generar visualizaciones comparativas y el reporte técnico final
a partir de los resultados de evaluación del Latam-Eval-Framework.

Métricas visualizadas:
    - ROUGE-L
    - BLEU
    - METEOR
    - Embedding Similarity (similitud semántica con BERT guaraní)
    - Cost-Performance Ratio (desempeño vs. costo/latencia estimados)

Uso:
    python generate_report.py
"""

import json

import matplotlib.pyplot as plt
import numpy as np


# ---------------------------------------------------------------------------
# Costos y latencias estimados por modelo
# (USD por millón de tokens de entrada, según precios públicos aprox.)
# ---------------------------------------------------------------------------
MODEL_COSTS = {
    "llama-3.1-8b-instant": 0.05,   # Groq: ~$0.05/M tokens (input)
    "mock": 0.0,                     # Sin costo real (simulado)
    "gemini-2.5-flash": 0.15,       # Google Gemini Flash aprox.
    "gpt-4o-mini": 0.15,            # OpenAI gpt-4o-mini
}

MODEL_LATENCY_MS = {
    "llama-3.1-8b-instant": 320,    # ~320ms promedio en Groq
    "mock": 500,                     # 0.5s de sleep simulado
    "gemini-2.5-flash": 1200,       # ~1200ms promedio
    "gpt-4o-mini": 800,             # ~800ms promedio
}

COLORS = ["#6C63FF", "#FF6584", "#43AA8B", "#F8961E", "#577590"]


def load_reports() -> dict:
    """
    Carga y combina los archivos de reporte JSON generados por el evaluador.

    Returns:
        dict: Diccionario combinado {model_name: {global_metrics, evaluations}}.
    """
    with open("report.json", "r", encoding="utf-8") as f:
        r1 = json.load(f)
    with open("report2.json", "r", encoding="utf-8") as f:
        r2 = json.load(f)

    r1.update(r2)
    return r1


def compute_avg_metrics(reports: dict) -> tuple:
    """
    Calcula los promedios de cada métrica por modelo.

    Args:
        reports (dict): Diccionario de reportes cargado por ``load_reports()``.

    Returns:
        tuple: (models, metrics) donde ``models`` es la lista de nombres y
               ``metrics`` es un dict con listas de valores promedio por métrica.
    """
    models = list(reports.keys())
    metrics = {
        "rougeL": [],
        "bleu": [],
        "meteor": [],
        "embedding_similarity": [],
    }

    for model in models:
        evals = reports[model]["evaluations"]
        metrics["rougeL"].append(
            np.mean([e["metrics"].get("rougeL", 0.0) for e in evals])
        )
        metrics["bleu"].append(
            np.mean([e["metrics"].get("bleu", 0.0) for e in evals])
        )
        metrics["meteor"].append(
            np.mean([e["metrics"].get("meteor", 0.0) for e in evals])
        )
        metrics["embedding_similarity"].append(
            np.mean([e["metrics"].get("embedding_similarity", 0.0) for e in evals])
        )

    return models, metrics


def plot_linguistic_metrics(models: list, metrics: dict) -> None:
    """
    Genera y guarda el gráfico de barras agrupadas con las cuatro métricas
    lingüísticas (ROUGE-L, BLEU, METEOR, Embedding Similarity).

    Args:
        models (list): Nombres de los modelos evaluados.
        metrics (dict): Promedios de métricas por modelo.
    """
    x = np.arange(len(models))
    width = 0.2

    fig, ax = plt.subplots(figsize=(12, 6))
    fig.patch.set_facecolor("#1A1A2E")
    ax.set_facecolor("#16213E")

    metric_keys = ["rougeL", "bleu", "meteor", "embedding_similarity"]
    metric_labels = ["ROUGE-L", "BLEU", "METEOR", "Emb. Similarity"]
    offsets = [-1.5, -0.5, 0.5, 1.5]

    bars = []
    for i, (key, label, offset) in enumerate(
        zip(metric_keys, metric_labels, offsets)
    ):
        b = ax.bar(
            x + offset * width,
            metrics[key],
            width,
            label=label,
            color=COLORS[i],
            alpha=0.9,
            edgecolor="white",
            linewidth=0.5,
        )
        bars.append(b)

    ax.set_ylabel("Score Promedio", color="white", fontsize=12)
    ax.set_title(
        "Comparativa de Métricas Lingüísticas por Modelo",
        color="white",
        fontsize=14,
        fontweight="bold",
        pad=15,
    )
    ax.set_xticks(x)
    ax.set_xticklabels(models, color="white", fontsize=11)
    ax.tick_params(axis="y", colors="white")
    ax.spines["bottom"].set_color("#555")
    ax.spines["left"].set_color("#555")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(axis="y", linestyle="--", alpha=0.3, color="white")
    ax.legend(facecolor="#0F3460", labelcolor="white", fontsize=10)
    ax.set_ylim(0, 1.05)

    plt.tight_layout()
    plt.savefig("comparativa_metricas.png", dpi=150, bbox_inches="tight")
    plt.close()
    print("✓ Guardado: comparativa_metricas.png")


def plot_cost_performance(models: list, metrics: dict) -> None:
    """
    Genera y guarda el gráfico de dispersión Cost-Performance Ratio.

    Ejes:
        - X: Costo estimado en USD por millón de tokens de entrada.
        - Y: Desempeño compuesto (promedio de METEOR + Embedding Similarity).

    Args:
        models (list): Nombres de los modelos evaluados.
        metrics (dict): Promedios de métricas por modelo.
    """
    costs = [MODEL_COSTS.get(m, 0.2) for m in models]
    # Performance compuesta: 60% METEOR + 40% Embedding Similarity
    performances = [
        0.6 * metrics["meteor"][i] + 0.4 * metrics["embedding_similarity"][i]
        for i in range(len(models))
    ]

    fig, ax = plt.subplots(figsize=(9, 6))
    fig.patch.set_facecolor("#1A1A2E")
    ax.set_facecolor("#16213E")

    for i, (model, cost, perf) in enumerate(zip(models, costs, performances)):
        color = COLORS[i % len(COLORS)]
        ax.scatter(cost, perf, s=220, color=color, zorder=5, edgecolors="white", linewidth=1.5)
        ax.annotate(
            model,
            (cost, perf),
            xytext=(10, 6),
            textcoords="offset points",
            color="white",
            fontsize=9,
            bbox=dict(boxstyle="round,pad=0.3", facecolor="#0F3460", alpha=0.7),
        )

    ax.set_xlabel(
        "Costo Estimado (USD / 1M tokens input)",
        color="white",
        fontsize=11,
    )
    ax.set_ylabel(
        "Desempeño Compuesto\n(0.6·METEOR + 0.4·Emb.Sim)",
        color="white",
        fontsize=11,
    )
    ax.set_title(
        "Cost-Performance Ratio: Desempeño vs. Costo",
        color="white",
        fontsize=14,
        fontweight="bold",
        pad=15,
    )
    ax.tick_params(colors="white")
    ax.spines["bottom"].set_color("#555")
    ax.spines["left"].set_color("#555")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(linestyle="--", alpha=0.3, color="white")

    plt.tight_layout()
    plt.savefig("cost_performance.png", dpi=150, bbox_inches="tight")
    plt.close()
    print("✓ Guardado: cost_performance.png")


def plot_latency_performance(models: list, metrics: dict) -> None:
    """
    Genera y guarda el gráfico de dispersión Latencia vs. Desempeño.

    Args:
        models (list): Nombres de los modelos evaluados.
        metrics (dict): Promedios de métricas por modelo.
    """
    latencies = [MODEL_LATENCY_MS.get(m, 1000) for m in models]
    performances = [
        0.6 * metrics["meteor"][i] + 0.4 * metrics["embedding_similarity"][i]
        for i in range(len(models))
    ]

    fig, ax = plt.subplots(figsize=(9, 6))
    fig.patch.set_facecolor("#1A1A2E")
    ax.set_facecolor("#16213E")

    for i, (model, lat, perf) in enumerate(zip(models, latencies, performances)):
        color = COLORS[i % len(COLORS)]
        ax.scatter(lat, perf, s=220, color=color, zorder=5, edgecolors="white", linewidth=1.5)
        ax.annotate(
            model,
            (lat, perf),
            xytext=(10, 6),
            textcoords="offset points",
            color="white",
            fontsize=9,
            bbox=dict(boxstyle="round,pad=0.3", facecolor="#0F3460", alpha=0.7),
        )

    ax.set_xlabel("Latencia Estimada (ms / request)", color="white", fontsize=11)
    ax.set_ylabel(
        "Desempeño Compuesto\n(0.6·METEOR + 0.4·Emb.Sim)",
        color="white",
        fontsize=11,
    )
    ax.set_title(
        "Latencia vs. Desempeño por Modelo",
        color="white",
        fontsize=14,
        fontweight="bold",
        pad=15,
    )
    ax.tick_params(colors="white")
    ax.spines["bottom"].set_color("#555")
    ax.spines["left"].set_color("#555")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(linestyle="--", alpha=0.3, color="white")

    plt.tight_layout()
    plt.savefig("latency_performance.png", dpi=150, bbox_inches="tight")
    plt.close()
    print("✓ Guardado: latency_performance.png")


def main() -> None:
    """
    Función principal. Carga los reportes y genera los tres gráficos comparativos:
        1. comparativa_metricas.png  — Todas las métricas lingüísticas por modelo.
        2. cost_performance.png      — Desempeño compuesto vs. Costo.
        3. latency_performance.png   — Desempeño compuesto vs. Latencia.
    """
    reports = load_reports()
    models, metrics = compute_avg_metrics(reports)

    print(f"\nModelos evaluados: {models}")
    for key, vals in metrics.items():
        for m, v in zip(models, vals):
            print(f"  [{m}] {key}: {v:.4f}")
    print()

    plot_linguistic_metrics(models, metrics)
    plot_cost_performance(models, metrics)
    plot_latency_performance(models, metrics)

    print("\n✅ Todas las visualizaciones generadas con éxito.")


if __name__ == "__main__":
    main()
