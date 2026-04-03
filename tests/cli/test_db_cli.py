from typer.testing import CliRunner

from providence.cli.app import app

runner = CliRunner()


def test_db_help_renders():
    result = runner.invoke(app, ["db", "--help"])
    assert result.exit_code == 0
    assert "backfill-payouts" in result.stdout
    assert "stats" in result.stdout
