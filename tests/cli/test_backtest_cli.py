from typer.testing import CliRunner

from providence.cli.app import app

runner = CliRunner()


def test_backtest_help_renders():
    result = runner.invoke(app, ["backtest", "--help"])
    assert result.exit_code == 0
    assert "judgment-time" in result.stdout
    assert "evaluation-mode" in result.stdout
    assert "ticket-types" in result.stdout
    assert "max-candidates" in result.stdout
    assert "fractional-kelly" in result.stdout
