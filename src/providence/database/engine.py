from sqlalchemy import create_engine, event
from sqlalchemy.engine import Engine
from sqlalchemy.orm import Session, sessionmaker

from providence.config import Settings

_engine: Engine | None = None
_session_factory: sessionmaker[Session] | None = None


def _enable_sqlite_fk(dbapi_conn, connection_record):  # noqa: ARG001
    cursor = dbapi_conn.cursor()
    cursor.execute("PRAGMA foreign_keys=ON")
    cursor.close()


def get_engine(settings: Settings | None = None) -> Engine:
    global _engine
    if _engine is not None:
        return _engine

    if settings is None:
        from providence.config import get_settings

        settings = get_settings()

    settings.ensure_data_dir()

    _engine = create_engine(
        settings.db_url,
        echo=False,
        pool_pre_ping=True,
    )

    event.listen(_engine, "connect", _enable_sqlite_fk)

    return _engine


def get_session_factory(settings: Settings | None = None) -> sessionmaker[Session]:
    global _session_factory
    if _session_factory is not None:
        return _session_factory

    engine = get_engine(settings)
    _session_factory = sessionmaker(bind=engine, expire_on_commit=False)
    return _session_factory


def reset_engine() -> None:
    """テスト用: グローバル状態をリセット"""
    global _engine, _session_factory
    if _engine is not None:
        _engine.dispose()
    _engine = None
    _session_factory = None
