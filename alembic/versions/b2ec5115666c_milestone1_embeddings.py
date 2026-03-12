"""milestone1 embeddings

Revision ID: <ALEMBIC_WILL_FILL_THIS>
Revises: be10eeb7f9d7
Create Date: 2026-03-02
"""

from __future__ import annotations

from collections.abc import Sequence

from alembic import op

# REQUIRED Alembic identifiers
revision: str = "<ALEMBIC_WILL_FILL_THIS>"
down_revision: str | None = "be10eeb7f9d7"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None

# Pick a dimension for your embeddings.
# 384 is a common default (e.g., many small sentence embedding models).
EMBED_DIM = 384


def upgrade() -> None:
    # Ensure pgvector extension exists (your container must support it)
    op.execute("CREATE EXTENSION IF NOT EXISTS vector;")

    # Add embedding column (vector)
    op.execute(
        f"ALTER TABLE doc_chunks ADD COLUMN IF NOT EXISTS embedding vector({EMBED_DIM});"
    )


def downgrade() -> None:
    op.execute("ALTER TABLE doc_chunks DROP COLUMN IF EXISTS embedding;")
