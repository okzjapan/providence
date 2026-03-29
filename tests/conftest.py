"""Shared test fixtures."""

import pytest
from sqlalchemy import create_engine, event
from sqlalchemy.orm import sessionmaker

from providence.database.tables import Base


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
