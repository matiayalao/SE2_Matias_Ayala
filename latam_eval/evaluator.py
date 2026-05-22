import json
import logging
from typing import List, Dict, Any

import nltk
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
from nltk.translate.meteor_score import single_meteor_score
from rouge_score import rouge_scorer
from sklearn.metrics import accuracy_score, f1_score

from latam_eval.models.base import BaseModelAdapter
from latam_eval.datasets.loader import DatasetLoader


class Evaluator:
    """
    Evaluator engine for Latam-Eval-Framework.
    Handles running datasets through models and calculating metrics.
    """

    def __init__(self, models: List[BaseModelAdapter], dataset_path: str):
        self.models = models
        self.dataset_loader = DatasetLoader(dataset_path)
        self.dataset = self.dataset_loader.load()
        self.scorer = rouge_scorer.RougeScorer(
            ["rouge1", "rouge2", "rougeL"], use_stemmer=True
        )

        # Ensure required NLTK data is downloaded
        self._ensure_nltk_resources()

    def _ensure_nltk_resources(self):
        """Downloads NLTK resources if not already present."""
        resources = [
            ("corpora/wordnet", "wordnet"),
            ("tokenizers/punkt", "punkt"),
            ("tokenizers/punkt_tab", "punkt_tab"),
        ]
        for res_path, res_name in resources:
            try:
                nltk.data.find(res_path)
            except LookupError:
                nltk.download(res_name, quiet=True)

    def evaluate(self) -> Dict[str, Any]:
        """
        Runs the evaluation pipeline.

        Returns:
            Dict[str, Any]: A dictionary containing results and metrics per model.
        """
        results = {}
        for model in self.models:
            logging.info(f"Evaluating model: {model.model_name}")
            model_results = []

            for item in self.dataset:
                instruction = item.get("instruction", "")
                expected = item.get("expected_response", "")

                try:
                    response = model.generate_response(instruction)
                except Exception as e:
                    logging.error(
                        f"Error generating response for {model.model_name}: {e}"
                    )
                    response = ""

                metrics = self._calculate_metrics(response, expected)

                model_results.append(
                    {
                        "id": item.get("id", "unknown"),
                        "instruction": instruction,
                        "expected": expected,
                        "response": response,
                        "metrics": metrics,
                    }
                )

            # Calcular F1 y Accuracy globales si es una tarea de clasificación
            y_true = []
            y_pred = []
            for res in model_results:
                exp = res["expected"].strip()
                resp = res["response"].strip()
                if exp in ["0", "1"]:
                    y_true.append(int(exp))
                    # Lógica simple para extraer 1 o 0 de la respuesta del modelo
                    pred_val = 1 if "1" in resp else 0
                    y_pred.append(pred_val)

            global_metrics = {}
            if len(y_true) > 0 and len(y_true) == len(model_results):
                global_metrics["accuracy"] = accuracy_score(y_true, y_pred)
                global_metrics["f1_macro"] = f1_score(y_true, y_pred, average="macro")
                logging.info(
                    f"[{model.model_name}] Accuracy: {global_metrics['accuracy']:.2f} | F1: {global_metrics['f1_macro']:.2f}"
                )

            results[model.model_name] = {
                "global_metrics": global_metrics,
                "evaluations": model_results,
            }

        return results

    def _calculate_metrics(self, hypothesis: str, reference: str) -> Dict[str, float]:
        """
        Calculates NLP metrics (ROUGE, BLEU, METEOR).
        """
        if not hypothesis or not reference:
            return {"rougeL": 0.0, "bleu": 0.0, "meteor": 0.0}

        # ROUGE
        rouge_scores = self.scorer.score(reference, hypothesis)
        rougeL = rouge_scores["rougeL"].fmeasure

        # BLEU
        ref_tokens = nltk.word_tokenize(reference)
        hyp_tokens = nltk.word_tokenize(hypothesis)
        smooth = SmoothingFunction().method1
        bleu = sentence_bleu([ref_tokens], hyp_tokens, smoothing_function=smooth)

        # METEOR
        try:
            meteor = single_meteor_score(ref_tokens, hyp_tokens)
        except Exception:
            meteor = 0.0

        return {"rougeL": rougeL, "bleu": bleu, "meteor": meteor}

    def save_report(self, results: Dict[str, Any], output_path: str):
        """
        Saves the evaluation results to a JSON file.
        """
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(results, f, ensure_ascii=False, indent=4)
        logging.info(f"Report successfully saved to {output_path}")
