import pytest
from latam_eval.cli import main, parse_model_names
from unittest.mock import patch

def test_cli_help(capsys):
    with patch('sys.argv', ['eval-llm', '--help']):
        with pytest.raises(SystemExit) as e:
            main()
        assert e.value.code == 0
        captured = capsys.readouterr()
        assert "Latam-Eval-Framework CLI" in captured.out


def test_parse_model_names_accepts_comma_separated_values_with_spaces():
    assert parse_model_names(["mock,", "gemini-2.5-flash,", "latamgpt"]) == [
        "mock",
        "gemini-2.5-flash",
        "latamgpt",
    ]


def test_parse_model_names_accepts_single_comma_separated_value():
    assert parse_model_names(["mock,gemini-2.5-flash,latamgpt"]) == [
        "mock",
        "gemini-2.5-flash",
        "latamgpt",
    ]
