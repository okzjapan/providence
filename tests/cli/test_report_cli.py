from datetime import date, datetime
from types import SimpleNamespace

from sqlalchemy import create_engine, event
from sqlalchemy.orm import sessionmaker
from typer.testing import CliRunner

from providence.cli.app import app
from providence.database.tables import Base, Race, ScrapeLog, StrategyRun, Track

runner = CliRunner()


def _enable_sqlite_fk(dbapi_conn, connection_record):  # noqa: ARG001
    cursor = dbapi_conn.cursor()
    cursor.execute("PRAGMA foreign_keys=ON")
    cursor.close()


def _session_factory():
    engine = create_engine("sqlite:///:memory:")
    event.listen(engine, "connect", _enable_sqlite_fk)
    Base.metadata.create_all(engine)
    return sessionmaker(bind=engine, expire_on_commit=False)


def test_report_cli_handles_empty_db(monkeypatch):
    factory = _session_factory()
    monkeypatch.setattr("providence.cli.report.get_session_factory", lambda: factory)
    monkeypatch.setattr("providence.cli.report.ModelStore.list_versions", lambda self: [])
    monkeypatch.setattr(
        "providence.cli.report.ModelStore.latest_version",
        lambda self: (_ for _ in ()).throw(FileNotFoundError()),
    )

    result = runner.invoke(app, ["report"])

    assert result.exit_code == 0
    assert "Strategy Operations" in result.stdout
    assert "Latest Model" in result.stdout
    assert "Recent Scrape Logs" in result.stdout

    refresh_result = runner.invoke(app, ["report", "--refresh"])
    assert refresh_result.exit_code == 0
    assert "Refresh Result" in refresh_result.stdout
    assert "Latest drift check" in refresh_result.stdout


def test_report_cli_renders_recent_activity(monkeypatch):
    factory = _session_factory()
    with factory() as session:
        session.add(Track(id=1, name="川口", location="埼玉県川口市"))
        session.add(Race(track_id=1, race_date=date(2026, 4, 2), race_number=1, grade="普通", distance=3100))
        session.flush()
        session.add(
            ScrapeLog(
                source="autorace_jp",
                target="odds",
                target_date=date(2026, 4, 2),
                records_count=100,
                status="success",
                duration_sec=3.2,
            )
        )
        session.add(
            StrategyRun(
                race_id=1,
                model_version="v003",
                evaluation_mode="live",
                judgment_time=datetime(2026, 4, 2, 10, 0, 0),
                total_recommended_bet=0.0,
                skip_reason="rounded_below_minimum",
            )
        )
        session.commit()

    monkeypatch.setattr("providence.cli.report.get_session_factory", lambda: factory)
    monkeypatch.setattr(
        "providence.cli.report.ModelStore.list_versions",
        lambda self: [
            {
                "version": "v003",
                "created_at": "2026-04-02T00:00:00+00:00",
                "model_type": "lambdarank",
                "metrics": {"win_accuracy": 0.45, "top3_overlap": 0.61, "brier_score": 0.09},
                "gate": {"passed": True},
            }
        ],
    )

    result = runner.invoke(app, ["report"])

    assert result.exit_code == 0
    assert "v003" in result.stdout
    assert "rounded_below_minimum" in result.stdout
    assert "autorace_jp" in result.stdout


def test_report_cli_refresh_runs_feedback(monkeypatch):
    factory = _session_factory()
    monkeypatch.setattr("providence.cli.report.get_session_factory", lambda: factory)
    monkeypatch.setattr("providence.cli.report.ModelStore.latest_metadata", lambda self: {"version": "v003", "metrics": {}, "gate": {}})
    monkeypatch.setattr("providence.cli.report.ModelStore.latest_version", lambda self: "v003")
    monkeypatch.setattr("providence.cli.report.Repository.get_recent_scrape_logs", lambda self, session, limit=10: [])
    monkeypatch.setattr(
        "providence.cli.report.Repository.get_strategy_run_summary",
        lambda self, session: {
            "strategy_runs": 0,
            "predictions": 0,
            "latest_prediction_at": None,
            "latest_run": None,
        },
    )
    monkeypatch.setattr("providence.cli.report.Repository.get_recent_model_performance", lambda self, session, limit=12, model_version=None: [])
    monkeypatch.setattr(
        "providence.cli.report.Repository.get_feedback_freshness",
        lambda self, session: {"latest_reconcile_at": None, "latest_performance_at": None, "latest_drift_at": None},
    )
    monkeypatch.setattr(
        "providence.cli.report.reconcile_paper_trades",
        lambda session, repository=None: SimpleNamespace(
            strategy_runs=1,
            betting_logs_written=2,
            reconciled_at=datetime(2025, 6, 15, 10, 0, 0),
        ),
    )
    monkeypatch.setattr(
        "providence.cli.report.refresh_model_performance",
        lambda session, end_date=None, repository=None: SimpleNamespace(
            rows_written=3,
            computed_at=datetime(2025, 6, 15, 11, 0, 0),
        ),
    )
    monkeypatch.setattr(
        "providence.cli.report.detect_drift",
        lambda session, model_version, evaluation_date, repository=None, store=None: SimpleNamespace(
            checked_at=evaluation_date,
            warnings=["roi_drop"],
            metrics={},
            psi_scores={},
        ),
    )

    result = runner.invoke(app, ["report", "--refresh"])
    assert result.exit_code == 0
    assert "Refresh Result" in result.stdout
    assert "Drift Check" in result.stdout
