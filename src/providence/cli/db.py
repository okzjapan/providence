"""Database management CLI commands."""

import typer
from rich.console import Console
from rich.table import Table

from providence.database.engine import get_session_factory
from providence.database.repository import Repository

db_app = typer.Typer()
console = Console()


@db_app.command("stats")
def db_stats() -> None:
    """Show database statistics."""
    session_factory = get_session_factory()
    repo = Repository()

    with session_factory() as session:
        repo.ensure_tracks(session)

    with session_factory() as session:
        stats = repo.get_db_stats(session)

    table = Table(title="Database Statistics")
    table.add_column("Table", style="cyan")
    table.add_column("Rows", justify="right", style="green")

    for name, count in stats.items():
        table.add_row(name, str(count))

    console.print(table)
