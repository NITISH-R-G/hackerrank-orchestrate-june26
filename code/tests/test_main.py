"""Tests for main.py entry point.

Verifies argument parsing, successful execution, and error handling.
"""
import pytest
import main


def test_main_default_args(mocker, caplog):
    """Test main() with default arguments successfully runs the pipeline."""
    mock_run_pipeline = mocker.patch("main.run_pipeline")

    # Passing empty list uses default args
    exit_code = main.main([])

    assert exit_code == 0
    mock_run_pipeline.assert_called_once()
    # Check that defaults were picked up correctly (from config or fallbacks)
    args, kwargs = mock_run_pipeline.call_args
    assert "claims_path" in kwargs
    assert "output_path" in kwargs


def test_main_custom_args(mocker, caplog):
    """Test main() correctly parses custom input and output paths."""
    mock_run_pipeline = mocker.patch("main.run_pipeline")

    custom_args = ["--input", "custom_input.csv", "--output", "custom_output.csv"]
    exit_code = main.main(custom_args)

    assert exit_code == 0
    mock_run_pipeline.assert_called_once_with(
        claims_path="custom_input.csv",
        output_path="custom_output.csv"
    )


def test_main_help_flag(capsys):
    """Test main() exits cleanly with code 0 when --help is passed."""
    exit_code = main.main(["--help"])

    assert exit_code == 0

    captured = capsys.readouterr()
    assert "Verify damage claims using multi-modal evidence." in captured.out
    assert "--input" in captured.out


def test_main_invalid_args(capsys):
    """Test main() exits with code 2 on unknown arguments."""
    exit_code = main.main(["--unknown-arg"])

    assert exit_code == 2

    captured = capsys.readouterr()
    assert "unrecognized arguments" in captured.err
    assert "--unknown-arg" in captured.err


def test_main_exception_handling(mocker, caplog):
    """Test main() catches unexpected exceptions from the pipeline and exits 1."""
    mock_run_pipeline = mocker.patch(
        "main.run_pipeline",
        side_effect=RuntimeError("Test exception")
    )

    exit_code = main.main([])

    assert exit_code == 1
    mock_run_pipeline.assert_called_once()
    assert "Pipeline failed: Test exception" in caplog.text
