"""Providence CLI entry point."""

import sys

import structlog
import typer

from providence.config import get_settings

app = typer.Typer(name="providence", help="Autorace prediction system")


def _configure_logging() -> None:
    settings = get_settings()
    structlog.configure(
        processors=[
            structlog.contextvars.merge_contextvars,
            structlog.processors.add_log_level,
            structlog.processors.TimeStamper(fmt="iso"),
            structlog.dev.ConsoleRenderer() if sys.stderr.isatty() else structlog.processors.JSONRenderer(),
        ],
        wrapper_class=structlog.make_filtering_bound_logger(
            {"DEBUG": 10, "INFO": 20, "WARNING": 30, "ERROR": 40, "CRITICAL": 50}.get(
                settings.log_level.upper(), 20
            )
        ),
        logger_factory=structlog.PrintLoggerFactory(file=sys.stderr),
        cache_logger_on_first_use=True,
    )


@app.callback()
def main() -> None:
    """Providence - Autorace prediction system."""
    _configure_logging()


# Register sub-command groups
from providence.cli.db import db_app  # noqa: E402
from providence.cli.model import model_app  # noqa: E402
from providence.cli.predict import predict_command  # noqa: E402
from providence.cli.scrape import scrape_app  # noqa: E402
from providence.cli.train import train_command  # noqa: E402

app.add_typer(scrape_app, name="scrape", help="Data collection commands")
app.add_typer(db_app, name="db", help="Database management commands")
app.add_typer(model_app, name="model", help="Model management commands")
app.command(name="train")(train_command)
app.command(name="predict")(predict_command)
