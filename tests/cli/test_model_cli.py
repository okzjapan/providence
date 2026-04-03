from typer.testing import CliRunner

from providence.cli.app import app

runner = CliRunner()


def test_model_help_renders():
    result = runner.invoke(app, ["model", "--help"])
    assert result.exit_code == 0
    assert "list" in result.stdout
    assert "evaluate" in result.stdout
