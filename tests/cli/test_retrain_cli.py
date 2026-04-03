from typer.testing import CliRunner

from providence.cli.app import app

runner = CliRunner()


def test_retrain_help_renders():
    result = runner.invoke(app, ["retrain", "--help"])
    assert result.exit_code == 0
    assert "compare-window-days" in result.stdout
    assert "--promote" in result.stdout
