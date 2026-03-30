import yaml
from pathlib import Path
from typing import Any, Dict


def load_config(file_path: str) -> Dict[str, Any]:
    """
    Loads an evaluation configuration from a YAML file.

    Args:
        file_path (str): The path to the config.yaml file.

    Returns:
        Dict[str, Any]: A dictionary containing the configuration parameters.

    Raises:
        FileNotFoundError: If the config file does not exist.
        ValueError: If the config file is not a valid YAML.
    """
    path = Path(file_path)
    if not path.exists():
        raise FileNotFoundError(f"Configuration file not found: {path}")

    with open(path, "r", encoding="utf-8") as f:
        try:
            config = yaml.safe_load(f)
        except yaml.YAMLError as e:
            raise ValueError(f"Failed to parse YAML configuration: {e}")

    return config or {}
