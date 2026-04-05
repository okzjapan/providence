"""Phase 5 feedback foundation.

Revision ID: 91a9f4f5d4c2
Revises: 7c1f2d4d9a6b
Create Date: 2026-04-02 12:30:00.000000
"""

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op

revision: str = "91a9f4f5d4c2"
down_revision: str | None = "7c1f2d4d9a6b"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    bind = op.get_bind()
    inspector = sa.inspect(bind)

    betting_columns = {column["name"] for column in inspector.get_columns("betting_log")}
    if "reconciled_at" not in betting_columns:
        op.add_column(
            "betting_log",
            sa.Column("reconciled_at", sa.DateTime(), server_default=sa.func.now(), nullable=False),
        )

    model_columns = {column["name"] for column in inspector.get_columns("model_performance")}
    if "computed_at" not in model_columns:
        op.add_column(
            "model_performance",
            sa.Column("computed_at", sa.DateTime(), server_default=sa.func.now(), nullable=False),
        )

    model_uniques = {tuple(item["column_names"]) for item in inspector.get_unique_constraints("model_performance")}
    if ("model_version", "evaluation_date", "window") not in model_uniques:
        op.create_unique_constraint(
            "uq_model_performance_identity",
            "model_performance",
            ["model_version", "evaluation_date", "window"],
        )


def downgrade() -> None:
    bind = op.get_bind()
    inspector = sa.inspect(bind)

    model_uniques = {tuple(item["column_names"]) for item in inspector.get_unique_constraints("model_performance")}
    model_columns = {column["name"] for column in inspector.get_columns("model_performance")}
    if ("model_version", "evaluation_date", "window") in model_uniques:
        op.drop_constraint("uq_model_performance_identity", "model_performance", type_="unique")

    if "computed_at" in model_columns:
        op.drop_column("model_performance", "computed_at")

    betting_columns = {column["name"] for column in inspector.get_columns("betting_log")}
    if "reconciled_at" in betting_columns:
        op.drop_column("betting_log", "reconciled_at")
