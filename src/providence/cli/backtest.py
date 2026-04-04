"""Backtest CLI command."""

from __future__ import annotations

import hashlib
from datetime import date, datetime

import typer
from rich.console import Console

from providence.backtest.engine import BacktestEngine
from providence.backtest.metrics import summarize_backtest
from providence.backtest.report import summary_table
from providence.backtest.types import BacktestRaceResult
from providence.cli.strategy_options import build_strategy_config
from providence.database.engine import get_session_factory
from providence.database.repository import Repository
from providence.database.tables import BettingLog, Prediction, SimulationRun, StrategyRun
from providence.domain.enums import TrackCode
from providence.strategy.normalize import format_combination
from providence.strategy.types import EvaluationMode

console = Console()


def backtest_command(
    from_date: str = typer.Option(..., "--from", help="Start date (YYYY-MM-DD)"),
    to_date: str = typer.Option(..., "--to", help="End date (YYYY-MM-DD)"),
    judgment_time: str = typer.Option(..., "--judgment-time", help="Daily cutoff time (HH:MM or HH:MM:SS)"),
    model_version: str = typer.Option("latest", "--model-version", help="Model version"),
    evaluation_mode: str = typer.Option("fixed", "--evaluation-mode", help="fixed or walk-forward"),
    track: str | None = typer.Option(None, "--track", help="Track name"),
    save: bool = typer.Option(False, "--save", help="Persist replay as simulation run"),
    use_final_odds: bool = typer.Option(False, "--use-final-odds", help="Use settled payout values as odds (post-hoc replay)"),
    ticket_types: str | None = typer.Option(None, "--ticket-types", help="Comma-separated ticket types to bet on (e.g. win,wide)"),
    max_candidates: int | None = typer.Option(None, "--max-candidates", help="Max candidate bets per race"),
    fractional_kelly: float | None = typer.Option(None, "--fractional-kelly", help="Kelly fraction multiplier (default 0.25)"),
) -> None:
    mode = EvaluationMode.FIXED if evaluation_mode == "fixed" else EvaluationMode.WALK_FORWARD
    track_code = TrackCode.from_name(track) if track else None
    config = build_strategy_config(
        ticket_types=ticket_types,
        max_candidates=max_candidates,
        fractional_kelly=fractional_kelly,
    )
    engine = BacktestEngine()
    start = date.fromisoformat(from_date)
    end = date.fromisoformat(to_date)
    results = engine.run(
        start_date=start,
        end_date=end,
        judgment_clock=_parse_clock(judgment_time),
        evaluation_mode=mode,
        model_version=model_version,
        track_id=track_code.value if track_code else None,
        use_final_odds=use_final_odds,
        config=config,
    )
    summary = summarize_backtest(results)
    console.print(summary_table(summary))

    if save:
        _persist_replay(
            results, summary,
            model_version=model_version, evaluation_mode=mode,
            start_date=start, end_date=end,
        )
        console.print("[green]Simulation run saved.[/green]")


def _persist_replay(
    results: list[BacktestRaceResult],
    summary,
    *,
    model_version: str,
    evaluation_mode: EvaluationMode,
    start_date: date,
    end_date: date,
) -> None:
    odds_policy = "final_closed"
    stake_sizing_rule = "min_100_normalized"
    semantic_key = _build_semantic_key(
        model_version=model_version,
        evaluation_mode=evaluation_mode.value,
        odds_policy=odds_policy,
        stake_sizing_rule=stake_sizing_rule,
        start_date=start_date,
        end_date=end_date,
    )

    session_factory = get_session_factory()
    repo = Repository()

    with session_factory() as session:
        sim_run = SimulationRun(
            semantic_key=semantic_key,
            model_version=model_version,
            evaluation_mode=evaluation_mode.value,
            odds_policy=odds_policy,
            stake_sizing_rule=stake_sizing_rule,
            date_range_start=start_date,
            date_range_end=end_date,
            total_races=summary.total_races,
            evaluated_races=summary.profit_evaluated_races,
            odds_missing_races=summary.accuracy_only_races,
            payout_missing_races=0,
            total_stake=summary.total_stake,
            total_payout=summary.total_payout,
            total_profit=summary.total_profit,
            roi=summary.roi,
            hit_rate=summary.hit_rate,
            status="completed",
        )
        repo.save_simulation_run(session, sim_run)

        with session.begin():
            for race_result in results:
                sr = race_result.strategy_result
                strategy_run = StrategyRun(
                    race_id=sr.race_id,
                    simulation_run_id=sim_run.id,
                    model_version=sr.model_version,
                    evaluation_mode=sr.decision_context.evaluation_mode.value,
                    judgment_time=sr.decision_context.judgment_time,
                    stake_sizing_rule=stake_sizing_rule,
                    confidence_score=sr.confidence_score,
                    skip_reason=sr.skip_reason,
                    total_recommended_bet=sr.total_recommended_bet,
                )
                session.add(strategy_run)
                session.flush()

                selected_lookup = {
                    (b.ticket_type, b.combination): b for b in sr.recommended_bets
                }
                for bet in (sr.candidate_bets or sr.recommended_bets):
                    bet_key = (bet.ticket_type, bet.combination)
                    selected = selected_lookup.get(bet_key)
                    prediction = Prediction(
                        race_id=sr.race_id,
                        strategy_run_id=strategy_run.id,
                        model_version=sr.model_version,
                        predicted_at=sr.decision_context.judgment_time,
                        ticket_type=bet.ticket_type.value,
                        combination=format_combination(bet.ticket_type, bet.combination),
                        predicted_prob=bet.probability,
                        odds_at_prediction=bet.odds_value,
                        expected_value=bet.expected_value,
                        kelly_fraction=bet.kelly_fraction,
                        stake_weight=bet.stake_weight,
                        recommended_bet=selected.recommended_bet if selected else 0.0,
                        confidence_score=bet.confidence_score,
                        skip_reason=bet.skip_reason if selected else sr.skip_reason,
                    )
                    session.add(prediction)
                    session.flush()

                    if selected and race_result.profit_evaluated:
                        settled_lookup = {
                            (s.recommendation.ticket_type, s.recommendation.combination): s
                            for s in race_result.settled_recommendations
                        }
                        settled_item = settled_lookup.get(bet_key)
                        actual_bet = selected.recommended_bet
                        payout_amount = settled_item.payout_amount if settled_item else 0.0
                        profit = settled_item.profit if settled_item else -actual_bet
                        session.add(BettingLog(
                            prediction_id=prediction.id,
                            race_id=sr.race_id,
                            actual_bet_amount=actual_bet,
                            payout=payout_amount,
                            profit=profit,
                        ))


def _build_semantic_key(**kwargs: object) -> str:
    raw = "|".join(f"{k}={v}" for k, v in sorted(kwargs.items()))
    return hashlib.sha256(raw.encode()).hexdigest()[:16]


def _parse_clock(value: str):
    for fmt in ("%H:%M:%S", "%H:%M"):
        try:
            return datetime.strptime(value, fmt).time()
        except ValueError:
            continue
    raise typer.BadParameter("judgment-time must be HH:MM or HH:MM:SS")
