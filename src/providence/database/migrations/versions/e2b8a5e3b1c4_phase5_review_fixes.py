"""Phase 5 review fixes.

Revision ID: e2b8a5e3b1c4
Revises: 91a9f4f5d4c2
Create Date: 2026-04-03 09:00:00.000000
"""

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op

revision: str = "e2b8a5e3b1c4"
down_revision: str | None = "91a9f4f5d4c2"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    bind = op.get_bind()
    inspector = sa.inspect(bind)

    if "feedback_runs" not in inspector.get_table_names():
        op.create_table(
            "feedback_runs",
            sa.Column("id", sa.Integer(), autoincrement=True, nullable=False),
            sa.Column("job_name", sa.String(), nullable=False),
            sa.Column("model_version", sa.String(), nullable=True),
            sa.Column("evaluation_date", sa.Date(), nullable=True),
            sa.Column("status", sa.String(), nullable=False),
            sa.Column("details", sa.String(), nullable=True),
            sa.Column("executed_at", sa.DateTime(), server_default=sa.func.now(), nullable=False),
            sa.PrimaryKeyConstraint("id"),
        )
        op.create_index("ix_feedback_runs_job_executed", "feedback_runs", ["job_name", "executed_at"], unique=False)


def downgrade() -> None:
    bind = op.get_bind()
    inspector = sa.inspect(bind)
    if "feedback_runs" in inspector.get_table_names():
        indexes = {index["name"] for index in inspector.get_indexes("feedback_runs")}
        if "ix_feedback_runs_job_executed" in indexes:
            op.drop_index("ix_feedback_runs_job_executed", table_name="feedback_runs")
        op.drop_table("feedback_runs")
