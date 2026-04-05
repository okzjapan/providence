"""Add race schedule columns.

Add scheduled_start_at, telvote_close_at, schedule_source,
schedule_fetched_at to races table for deadline-based notification.

Revision ID: b4f8e2a1c7d3
Revises: a3d7c9e1f5b2
Create Date: 2026-04-04 12:00:00.000000
"""

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op

revision: str = "b4f8e2a1c7d3"
down_revision: str | None = "a3d7c9e1f5b2"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    bind = op.get_bind()
    inspector = sa.inspect(bind)
    existing = {c["name"] for c in inspector.get_columns("races")}

    if "scheduled_start_at" not in existing:
        op.add_column("races", sa.Column("scheduled_start_at", sa.DateTime(), nullable=True))
    if "telvote_close_at" not in existing:
        op.add_column("races", sa.Column("telvote_close_at", sa.DateTime(), nullable=True))
    if "schedule_source" not in existing:
        op.add_column("races", sa.Column("schedule_source", sa.String(), nullable=True))
    if "schedule_fetched_at" not in existing:
        op.add_column("races", sa.Column("schedule_fetched_at", sa.DateTime(), nullable=True))

    existing_indexes = {idx["name"] for idx in inspector.get_indexes("races")}
    if "ix_races_telvote_close" not in existing_indexes:
        op.create_index("ix_races_telvote_close", "races", ["telvote_close_at"])


def downgrade() -> None:
    op.drop_index("ix_races_telvote_close", table_name="races")
    op.drop_column("races", "schedule_fetched_at")
    op.drop_column("races", "schedule_source")
    op.drop_column("races", "telvote_close_at")
    op.drop_column("races", "scheduled_start_at")
