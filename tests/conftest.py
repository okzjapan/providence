"""Shared test fixtures."""

import io

import pytest
import structlog
from sqlalchemy import create_engine, event
from sqlalchemy.orm import sessionmaker

from providence.database.tables import Base
import providence.keiba.database.tables  # noqa: F401  -- register keiba tables with Base.metadata

@pytest.fixture(autouse=True)
def _reset_structlog():
    """Ensure structlog writes to a safe target after each test.

    CLI tests invoke _configure_logging() which binds structlog to stderr.
    CliRunner may close stderr, leaving cached loggers with a dead file handle.
    """
    yield
    structlog.configure(
        processors=[structlog.dev.ConsoleRenderer()],
        wrapper_class=structlog.make_filtering_bound_logger(40),
        logger_factory=structlog.PrintLoggerFactory(file=io.StringIO()),
        cache_logger_on_first_use=False,
    )


def _enable_sqlite_fk(dbapi_conn, connection_record):  # noqa: ARG001
    cursor = dbapi_conn.cursor()
    cursor.execute("PRAGMA foreign_keys=ON")
    cursor.close()


@pytest.fixture()
def session_factory():
    """In-memory SQLite session factory with foreign keys enabled.

    Each test gets a fresh database. Use session_factory() to create sessions.
    """
    engine = create_engine("sqlite:///:memory:")
    event.listen(engine, "connect", _enable_sqlite_fk)
    Base.metadata.create_all(engine)

    factory = sessionmaker(bind=engine, expire_on_commit=False)
    yield factory
    engine.dispose()
