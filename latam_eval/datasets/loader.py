import json
import csv
from pathlib import Path
from typing import List, Dict, Any


class DatasetLoader:
    """
    Handles loading and parsing of datasets for the evaluation framework.
    Supports JSON and CSV formats.
    """

    def __init__(self, file_path: str):
        self.file_path = Path(file_path)

    def load(self) -> List[Dict[str, Any]]:
        """
        Loads the dataset from the specified file.
        """
        if not self.file_path.exists():
            raise FileNotFoundError(f"Dataset file not found: {self.file_path}")

        ext = self.file_path.suffix.lower()
        if ext == ".json":
            return self._load_json()
        elif ext == ".csv":
            return self._load_csv()
        else:
            raise ValueError(f"Formato {ext} no soportado.")

    def _load_json(self) -> List[Dict[str, Any]]:
        with open(self.file_path, "r", encoding="utf-8") as f:
            try:
                data = json.load(f)
            except json.JSONDecodeError as e:
                raise ValueError(f"Failed to parse JSON dataset: {e}")
        return data

    def _load_csv(self) -> List[Dict[str, Any]]:
        data = []
        with open(self.file_path, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                # Detección del formato WNLI para clasificación
                if "sentence1" in row and "sentence2" in row and "score" in row:
                    instruction = (
                        "Analizá si la Oración 2 se deduce o es consecuencia lógica "
                        "de la Oración 1 (Entailment).\n"
                        f"Oración 1: {row['sentence1']}\n"
                        f"Oración 2: {row['sentence2']}\n"
                        "Respondé SOLAMENTE con un '1' si es correcto (se deduce) "
                        "o '0' si es incorrecto (no se deduce)."
                    )
                    data.append(
                        {
                            "id": row.get("ref", ""),
                            "instruction": instruction,
                            "expected_response": str(row["score"]).strip(),
                        }
                    )
                else:
                    data.append(row)
        return data
