import pytest
import sys
from main import parse_args, main

def test_parse_args_defaults():
    args = parse_args([])
    assert args.input == "dataset/claims.csv"
    assert args.output == "output.csv"
    assert args.sample is False

def test_parse_args_custom_files():
    args = parse_args(["--input", "custom_in.csv", "--output", "custom_out.csv"])
    assert args.input == "custom_in.csv"
    assert args.output == "custom_out.csv"
    assert args.sample is False

def test_parse_args_sample_flag():
    args = parse_args(["--sample"])
    assert args.sample is True

def test_parse_args_help(capsys):
    with pytest.raises(SystemExit) as excinfo:
        parse_args(["--help"])
    assert excinfo.value.code == 0
    captured = capsys.readouterr()
    assert "Multi-Modal Evidence Review System" in captured.out

def test_parse_args_invalid_argument(capsys):
    with pytest.raises(SystemExit) as excinfo:
        parse_args(["--invalid-flag"])
    assert excinfo.value.code == 2
    captured = capsys.readouterr()
    assert "unrecognized arguments: --invalid-flag" in captured.err

def test_main_execution_defaults(capsys):
    exit_code = main([])
    assert exit_code == 0
    captured = capsys.readouterr()
    assert "Starting evidence review pipeline" in captured.out
    assert "Input: dataset/claims.csv" in captured.out
    assert "Output: output.csv" in captured.out

def test_main_execution_sample_flag(capsys):
    exit_code = main(["--sample"])
    assert exit_code == 0
    captured = capsys.readouterr()
    assert "Input: dataset/sample_claims.csv" in captured.out

def test_main_execution_custom_args(capsys):
    exit_code = main(["--input", "test_in.csv", "--output", "test_out.csv"])
    assert exit_code == 0
    captured = capsys.readouterr()
    assert "Input: test_in.csv" in captured.out
    assert "Output: test_out.csv" in captured.out

def test_main_help_returns_zero(capsys):
    exit_code = main(["--help"])
    assert exit_code == 0

def test_main_invalid_args_returns_error(capsys):
    exit_code = main(["--invalid"])
    assert exit_code == 2
