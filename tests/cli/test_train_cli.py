"""CLI tests for train command argument handling."""

from typer.testing import CliRunner

from providence.cli.app import app

runner = CliRunner()


def test_train_requires_both_manual_split_dates():
    result = runner.invoke(app, ["train", "--train-end", "2024-12-31"])
    assert result.exit_code == 1
    assert "両方指定するか、両方省略" in result.stdout
