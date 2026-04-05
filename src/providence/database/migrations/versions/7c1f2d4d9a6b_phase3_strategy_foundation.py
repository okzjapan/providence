"""Phase 3 strategy foundation.

Revision ID: 7c1f2d4d9a6b
Revises: 56eeb72c2be3
Create Date: 2026-04-02 12:00:00.000000
"""

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op

# revision identifiers, used by Alembic.
revision: str = "7c1f2d4d9a6b"
down_revision: str | None = "56eeb72c2be3"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    """Upgrade schema."""
    bind = op.get_bind()
    inspector = sa.inspect(bind)

    odds_columns = {column["name"] for column in inspector.get_columns("odds_snapshot")}
    if "ingestion_batch_id" not in odds_columns:
        op.add_column("odds_snapshot", sa.Column("ingestion_batch_id", sa.String(), nullable=True))
    if "source_name" not in odds_columns:
        op.add_column("odds_snapshot", sa.Column("source_name", sa.String(), nullable=True))
    odds_indexes = {index["name"] for index in inspector.get_indexes("odds_snapshot")}
    if "ix_odds_race_batch" not in odds_indexes:
        op.create_index("ix_odds_race_batch", "odds_snapshot", ["race_id", "ingestion_batch_id"], unique=False)

    if "ticket_payouts" not in inspector.get_table_names():
        op.create_table(
            "ticket_payouts",
            sa.Column("id", sa.Integer(), autoincrement=True, nullable=False),
            sa.Column("race_id", sa.Integer(), nullable=False),
            sa.Column("ticket_type", sa.String(), nullable=False),
            sa.Column("combination", sa.String(), nullable=False),
            sa.Column("payout_value", sa.Float(), nullable=False),
            sa.Column("popularity", sa.Integer(), nullable=True),
            sa.Column("settled_at", sa.DateTime(), server_default=sa.func.now(), nullable=False),
            sa.ForeignKeyConstraint(["race_id"], ["races.id"]),
            sa.PrimaryKeyConstraint("id"),
            sa.UniqueConstraint("race_id", "ticket_type", "combination", name="uq_ticket_payout_identity"),
        )
        op.create_index("ix_ticket_payout_race_type", "ticket_payouts", ["race_id", "ticket_type"], unique=False)

    if "strategy_runs" not in inspector.get_table_names():
        op.create_table(
            "strategy_runs",
            sa.Column("id", sa.Integer(), autoincrement=True, nullable=False),
            sa.Column("race_id", sa.Integer(), nullable=False),
            sa.Column("model_version", sa.String(), nullable=False),
            sa.Column("evaluation_mode", sa.String(), nullable=False),
            sa.Column("judgment_time", sa.DateTime(), nullable=False),
            sa.Column("bankroll_before", sa.Float(), nullable=True),
            sa.Column("bankroll_after", sa.Float(), nullable=True),
            sa.Column("race_cap_fraction", sa.Float(), nullable=True),
            sa.Column("confidence_score", sa.Float(), nullable=True),
            sa.Column("skip_reason", sa.String(), nullable=True),
            sa.Column("total_recommended_bet", sa.Float(), nullable=False, server_default="0"),
            sa.Column("created_at", sa.DateTime(), server_default=sa.func.now(), nullable=False),
            sa.ForeignKeyConstraint(["race_id"], ["races.id"]),
            sa.PrimaryKeyConstraint("id"),
        )
        op.create_index("ix_strategy_run_race_judgment", "strategy_runs", ["race_id", "judgment_time"], unique=False)

    prediction_columns = {column["name"] for column in inspector.get_columns("predictions")}
    prediction_foreign_keys = {tuple(fk["constrained_columns"]) for fk in inspector.get_foreign_keys("predictions")}
    if "strategy_run_id" not in prediction_columns:
        op.add_column("predictions", sa.Column("strategy_run_id", sa.Integer(), nullable=True))
    if "skip_reason" not in prediction_columns:
        op.add_column("predictions", sa.Column("skip_reason", sa.String(), nullable=True))
    if ("strategy_run_id",) not in prediction_foreign_keys:
        op.create_foreign_key(
            "fk_predictions_strategy_run_id",
            "predictions",
            "strategy_runs",
            ["strategy_run_id"],
            ["id"],
        )


def downgrade() -> None:
    """Downgrade schema."""
    bind = op.get_bind()
    inspector = sa.inspect(bind)

    if "predictions" in inspector.get_table_names():
        prediction_columns = {column["name"] for column in inspector.get_columns("predictions")}
        prediction_foreign_keys = {tuple(fk["constrained_columns"]) for fk in inspector.get_foreign_keys("predictions")}
        if ("strategy_run_id",) in prediction_foreign_keys:
            op.drop_constraint("fk_predictions_strategy_run_id", "predictions", type_="foreignkey")
        if "skip_reason" in prediction_columns:
            op.drop_column("predictions", "skip_reason")
        if "strategy_run_id" in prediction_columns:
            op.drop_column("predictions", "strategy_run_id")

    if "strategy_runs" in inspector.get_table_names():
        strategy_indexes = {index["name"] for index in inspector.get_indexes("strategy_runs")}
        if "ix_strategy_run_race_judgment" in strategy_indexes:
            op.drop_index("ix_strategy_run_race_judgment", table_name="strategy_runs")
        op.drop_table("strategy_runs")

    if "ticket_payouts" in inspector.get_table_names():
        payout_indexes = {index["name"] for index in inspector.get_indexes("ticket_payouts")}
        if "ix_ticket_payout_race_type" in payout_indexes:
            op.drop_index("ix_ticket_payout_race_type", table_name="ticket_payouts")
        op.drop_table("ticket_payouts")

    if "odds_snapshot" in inspector.get_table_names():
        odds_indexes = {index["name"] for index in inspector.get_indexes("odds_snapshot")}
        odds_columns = {column["name"] for column in inspector.get_columns("odds_snapshot")}
        if "ix_odds_race_batch" in odds_indexes:
            op.drop_index("ix_odds_race_batch", table_name="odds_snapshot")
        if "source_name" in odds_columns:
            op.drop_column("odds_snapshot", "source_name")
        if "ingestion_batch_id" in odds_columns:
            op.drop_column("odds_snapshot", "ingestion_batch_id")
