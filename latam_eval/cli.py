import argparse
import logging
import sys
import os

from latam_eval.evaluator import Evaluator
from latam_eval.models.mock_adapter import MockAdapter
from latam_eval.models.openai_adapter import OpenAIAdapter
from latam_eval.models.gemini_adapter import GeminiAdapter
from latam_eval.models.universal_api_adapter import UniversalOpenAIAdapter
from latam_eval.utils.config import load_config


def main():
    parser = argparse.ArgumentParser(description="Latam-Eval-Framework CLI")
    parser.add_argument(
        "--config",
        type=str,
        help="Path to the YAML configuration file",
    )
    parser.add_argument(
        "--models",
        type=str,
        help="Comma-separated list of models (e.g. gpt-4o,gemini-1.5-flash,mock)",
    )
    parser.add_argument(
        "--dataset", type=str, help="Path to the JSON dataset file"
    )
    parser.add_argument(
        "--output",
        type=str,
        default="results.json",
        help="Path to save the output report",
    )

    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

    generation_params = {}
    model_names = []
    dataset_path = ""

    if args.config:
        try:
            config = load_config(args.config)
            if "models" in config:
                model_names = config["models"]
            if "dataset" in config:
                dataset_path = config["dataset"]
            if "parameters" in config:
                generation_params = config["parameters"]
            if "output" in config and args.output == "results.json":
                args.output = config["output"]
            logging.info(f"Loaded config from {args.config}")
        except Exception as e:
            logging.error(f"Error loading config: {e}")
            sys.exit(1)
    else:
        if args.models:
            model_names = [m.strip() for m in args.models.split(",")]
        if args.dataset:
            dataset_path = args.dataset

    if not model_names or not dataset_path:
        logging.error("You must provide either a --config file or both --models and --dataset.")
        sys.exit(1)
    models = []

    for m in model_names:
        lower_m = m.lower()
        if "mock" in lower_m:
            models.append(MockAdapter(m, **generation_params))
        elif "gpt" in lower_m:
            if not os.getenv("OPENAI_API_KEY"):
                logging.error(f"OPENAI_API_KEY not set for model {m}")
                sys.exit(1)
            models.append(OpenAIAdapter(m, **generation_params))
        elif "gemini" in lower_m:
            if not os.getenv("GEMINI_API_KEY"):
                logging.error(f"GEMINI_API_KEY not set for model {m}")
                sys.exit(1)
            models.append(GeminiAdapter(m, **generation_params))
        elif "llama" in lower_m or "grok" in lower_m:
            if not os.getenv("GROQ_API_KEY"):
                logging.error(f"GROQ_API_KEY not set for model {m}. Required for Llama/Grok via Groq.")
                sys.exit(1)
            models.append(
                UniversalOpenAIAdapter(
                    m,
                    base_url="https://api.groq.com/openai/v1",
                    api_key_env_var="GROQ_API_KEY",
                    **generation_params
                )
            )
        elif "local" in lower_m:
            logging.info(
                f"Conectando a modelo local usando LM Studio / Ollama en puerto 1234"
            )
            os.environ["LOCAL_DUMMY_KEY"] = "lm-studio"
            models.append(
                UniversalOpenAIAdapter(
                    "local-gemma",
                    base_url="http://localhost:1234/v1",
                    api_key_env_var="LOCAL_DUMMY_KEY",
                    **generation_params
                )
            )
        else:
            logging.warning(f"Unknown model '{m}', falling back to MockAdapter.")
            models.append(MockAdapter(f"Mock-{m}", **generation_params))

    if not models:
        logging.error("No valid models initialized.")
        sys.exit(1)

    evaluator = Evaluator(models, dataset_path)
    results = evaluator.evaluate()
    evaluator.save_report(results, args.output)


if __name__ == "__main__":
    main()
