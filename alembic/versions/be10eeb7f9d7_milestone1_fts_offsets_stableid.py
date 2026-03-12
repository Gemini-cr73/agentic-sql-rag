"""milestone1 fts offsets stableid

Revision ID: be10eeb7f9d7
Revises: 1ad82ce49884
Create Date: 2026-03-02
"""

from __future__ import annotations

from collections.abc import Sequence

import sqlalchemy as sa
from sqlalchemy.dialects import postgresql

from alembic import op

# ✅ Alembic identifiers (REQUIRED)
revision: str = "be10eeb7f9d7"
down_revision: str | None = "1ad82ce49884"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    # ------------------------------------------------------------
    # 0) Ensure pgcrypto exists (needed for digest())
    # ------------------------------------------------------------
    op.execute("CREATE EXTENSION IF NOT EXISTS pgcrypto;")

    # ------------------------------------------------------------
    # 1) Add new columns to doc_chunks
    # ------------------------------------------------------------
    op.add_column(
        "doc_chunks", sa.Column("stable_id", sa.String(length=64), nullable=True)
    )

    op.add_column(
        "doc_chunks",
        sa.Column(
            "char_start", sa.Integer(), nullable=False, server_default=sa.text("0")
        ),
    )
    op.add_column(
        "doc_chunks",
        sa.Column(
            "char_end", sa.Integer(), nullable=False, server_default=sa.text("0")
        ),
    )

    # TSVECTOR column for FTS
    op.add_column("doc_chunks", sa.Column("tsv", postgresql.TSVECTOR(), nullable=True))

    # ------------------------------------------------------------
    # 2) Backfill stable_id for existing rows (deterministic using id)
    # ------------------------------------------------------------
    op.execute(
        """
        UPDATE doc_chunks
        SET stable_id = encode(digest(id::text, 'sha256'), 'hex')
        WHERE stable_id IS NULL;
        """
    )

    # Make stable_id required + unique index
    op.alter_column("doc_chunks", "stable_id", nullable=False)
    op.create_index("ix_doc_chunks_stable_id", "doc_chunks", ["stable_id"], unique=True)

    # ------------------------------------------------------------
    # 3) Trigger/function to keep tsv updated
    # ------------------------------------------------------------
    op.execute(
        """
        CREATE OR REPLACE FUNCTION doc_chunks_tsv_update() RETURNS trigger AS $$
        BEGIN
          NEW.tsv := to_tsvector('english', COALESCE(NEW.text, ''));
          RETURN NEW;
        END
        $$ LANGUAGE plpgsql;
        """
    )

    op.execute(
        """
        DROP TRIGGER IF EXISTS trg_doc_chunks_tsv ON doc_chunks;
        CREATE TRIGGER trg_doc_chunks_tsv
        BEFORE INSERT OR UPDATE ON doc_chunks
        FOR EACH ROW EXECUTE FUNCTION doc_chunks_tsv_update();
        """
    )

    # Backfill tsv
    op.execute(
        "UPDATE doc_chunks SET tsv = to_tsvector('english', COALESCE(text, ''));"
    )

    # GIN index for FTS (use Alembic helper, avoids quoting issues)
    op.create_index(
        "ix_doc_chunks_tsv_gin",
        "doc_chunks",
        ["tsv"],
        unique=False,
        postgresql_using="gin",
    )

    # Optional: remove server defaults going forward
    op.alter_column("doc_chunks", "char_start", server_default=None)
    op.alter_column("doc_chunks", "char_end", server_default=None)


def downgrade() -> None:
    op.drop_index("ix_doc_chunks_tsv_gin", table_name="doc_chunks")
    op.execute("DROP TRIGGER IF EXISTS trg_doc_chunks_tsv ON doc_chunks;")
    op.execute("DROP FUNCTION IF EXISTS doc_chunks_tsv_update;")

    op.drop_index("ix_doc_chunks_stable_id", table_name="doc_chunks")

    op.drop_column("doc_chunks", "tsv")
    op.drop_column("doc_chunks", "char_end")
    op.drop_column("doc_chunks", "char_start")
    op.drop_column("doc_chunks", "stable_id")
