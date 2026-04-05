"""Simulation runs and stake normalization.

Add simulation_runs table, add simulation_run_id / stake_sizing_rule to
strategy_runs, add stake_weight to predictions.

Revision ID: a3d7c9e1f5b2
Revises: e2b8a5e3b1c4
Create Date: 2026-04-04 12:00:00.000000
"""

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op

revision: str = "a3d7c9e1f5b2"
down_revision: str | None = "e2b8a5e3b1c4"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    bind = op.get_bind()
    inspector = sa.inspect(bind)

    if "simulation_runs" not in inspector.get_table_names():
        op.create_table(
            "simulation_runs",
            sa.Column("id", sa.Integer(), autoincrement=True, nullable=False),
            sa.Column("semantic_key", sa.String(), nullable=False),
            sa.Column("model_version", sa.String(), nullable=False),
            sa.Column("evaluation_mode", sa.String(), nullable=False),
            sa.Column("odds_policy", sa.String(), nullable=False, server_default="final_closed"),
            sa.Column("stake_sizing_rule", sa.String(), nullable=False, server_default="min_100_normalized"),
            sa.Column("date_range_start", sa.Date(), nullable=False),
            sa.Column("date_range_end", sa.Date(), nullable=False),
            sa.Column("total_races", sa.Integer(), nullable=False, server_default="0"),
            sa.Column("evaluated_races", sa.Integer(), nullable=False, server_default="0"),
            sa.Column("odds_missing_races", sa.Integer(), nullable=False, server_default="0"),
            sa.Column("payout_missing_races", sa.Integer(), nullable=False, server_default="0"),
            sa.Column("total_stake", sa.Float(), nullable=False, server_default="0"),
            sa.Column("total_payout", sa.Float(), nullable=False, server_default="0"),
            sa.Column("total_profit", sa.Float(), nullable=False, server_default="0"),
            sa.Column("roi", sa.Float(), nullable=False, server_default="0"),
            sa.Column("hit_rate", sa.Float(), nullable=False, server_default="0"),
            sa.Column("status", sa.String(), nullable=False, server_default="completed"),
            sa.Column("created_at", sa.DateTime(), server_default=sa.func.now(), nullable=False),
            sa.PrimaryKeyConstraint("id"),
        )

    strategy_columns = {c["name"] for c in inspector.get_columns("strategy_runs")}
    if "simulation_run_id" not in strategy_columns:
        op.add_column("strategy_runs", sa.Column("simulation_run_id", sa.Integer(), nullable=True))
        op.create_foreign_key(
            "fk_strategy_runs_simulation_run_id",
            "strategy_runs",
            "simulation_runs",
            ["simulation_run_id"],
            ["id"],
        )
    if "stake_sizing_rule" not in strategy_columns:
        op.add_column("strategy_runs", sa.Column("stake_sizing_rule", sa.String(), nullable=True))

    prediction_columns = {c["name"] for c in inspector.get_columns("predictions")}
    if "stake_weight" not in prediction_columns:
        op.add_column("predictions", sa.Column("stake_weight", sa.Float(), nullable=True))


def downgrade() -> None:
    bind = op.get_bind()
    inspector = sa.inspect(bind)

    prediction_columns = {c["name"] for c in inspector.get_columns("predictions")}
    if "stake_weight" in prediction_columns:
        op.drop_column("predictions", "stake_weight")

    strategy_columns = {c["name"] for c in inspector.get_columns("strategy_runs")}
    if "simulation_run_id" in strategy_columns:
        op.drop_constraint("fk_strategy_runs_simulation_run_id", "strategy_runs", type_="foreignkey")
        op.drop_column("strategy_runs", "simulation_run_id")
    if "stake_sizing_rule" in strategy_columns:
        op.drop_column("strategy_runs", "stake_sizing_rule")

    if "simulation_runs" in inspector.get_table_names():
        op.drop_table("simulation_runs")
