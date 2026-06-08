import pytest
from latam_eval.evaluator import Evaluator
from latam_eval.models.mock_adapter import MockAdapter
import os
import json

def test_evaluator_mock_model(tmp_path):
    dataset_file = tmp_path / "test_dataset.json"
    dataset_file.write_text(json.dumps([
        {
            "id": "q1",
            "category": "traduccion",
            "instruction": "Traduce 'Hola' al guaraní.",
            "expected_response": "Mba'éichapa"
        }
    ]))
    
    mock_model = MockAdapter("mock-test")
    evaluator = Evaluator([mock_model], str(dataset_file))
    
    results = evaluator.evaluate()
    
    assert "mock-test" in results
    assert len(results["mock-test"]["evaluations"]) == 1
    assert results["mock-test"]["evaluations"][0]["id"] == "q1"
    assert "response" in results["mock-test"]["evaluations"][0]
