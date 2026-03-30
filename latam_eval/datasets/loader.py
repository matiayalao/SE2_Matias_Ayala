import json
from pathlib import Path
from typing import List, Dict, Any


class DatasetLoader:
    """
    Handles loading and parsing of datasets for the evaluation framework.
    Currently supports JSON formats with instruction-response pairs.
    """

    def __init__(self, file_path: str):
        """
        Initializes the DatasetLoader with a file path.

        Args:
            file_path (str): The path to the dataset file (JSON).
        """
        self.file_path = Path(file_path)

    def load(self) -> List[Dict[str, Any]]:
        """
        Loads the dataset from the specified file.

        Returns:
            List[Dict[str, Any]]: A list of dataset records, each containing
            at least 'instruction' and 'expected_response' (if supervised).

        Raises:
            FileNotFoundError: If the dataset file does not exist.
            ValueError: If the dataset is not a valid JSON or has the wrong format.
        """
        if not self.file_path.exists():
            raise FileNotFoundError(f"Dataset file not found: {self.file_path}")

        if self.file_path.suffix.lower() != ".json":
            raise ValueError("Currently, only .json format is supported.")

        with open(self.file_path, "r", encoding="utf-8") as f:
            try:
                data = json.load(f)
            except json.JSONDecodeError as e:
                raise ValueError(f"Failed to parse JSON dataset: {e}")

        if not isinstance(data, list):
            raise ValueError("Dataset JSON should contain a top-level list of objects.")

        # Basic validation of keys could be added here
        return data
