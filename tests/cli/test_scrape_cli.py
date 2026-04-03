from typer.testing import CliRunner

from providence.cli.app import app

runner = CliRunner()


def test_scrape_help_lists_results_command():
    result = runner.invoke(app, ["scrape", "--help"])
    assert result.exit_code == 0
    assert "results" in result.stdout
