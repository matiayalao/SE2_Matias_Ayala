import pytest
from latam_eval.cli import main
from unittest.mock import patch

def test_cli_help(capsys):
    with patch('sys.argv', ['eval-llm', '--help']):
        with pytest.raises(SystemExit) as e:
            main()
        assert e.value.code == 0
        captured = capsys.readouterr()
        assert "Latam-Eval-Framework CLI" in captured.out
