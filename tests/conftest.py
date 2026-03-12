# tests/conftest.py
from __future__ import annotations

import os
import pathlib
import uuid

import pytest
from sqlalchemy import text

from app.db.session import new_session


def _load_env_file_if_needed() -> None:
    """
    Ensure DATABASE_URL is available when running pytest directly.
    Loads docker/.env into *process* env if DATABASE_URL is missing.
    """
    if os.getenv("DATABASE_URL"):
        return

    root = pathlib.Path(__file__).resolve().parents[1]
    env_path = root / "docker" / ".env"
    if not env_path.exists():
        return

    for line in env_path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        k, v = line.split("=", 1)
        k, v = k.strip(), v.strip().strip('"').strip("'")
        os.environ.setdefault(k, v)


@pytest.fixture(scope="session", autouse=True)
def _ensure_env_loaded() -> None:
    _load_env_file_if_needed()


@pytest.fixture()
def seeded_doc_and_chunk() -> dict[str, int]:
    """
    Inserts 1 document + 1 chunk so retrieval has something to return.
    IMPORTANT: documents.content is NOT NULL in your schema, so we must provide it.
    """
    suffix = uuid.uuid4().hex[:8]
    title = f"pytest-title-{suffix}"
    source = f"pytest:{suffix}"
    content = f"pytest document content {suffix}"

    chunk_text = "This is a test chunk about retrieval and hybrid search."
    stable_id = f"pytest-{suffix}-0000"

    with new_session() as s:
        # Insert document (content is required)
        doc_id = s.execute(
            text(
                """
                INSERT INTO documents (title, source, content)
                VALUES (:title, :source, :content)
                RETURNING id
                """
            ),
            {"title": title, "source": source, "content": content},
        ).scalar_one()

        # Insert chunk
        chunk_id = s.execute(
            text(
                """
                INSERT INTO doc_chunks (
                    document_id, stable_id, chunk_index,
                    char_start, char_end, text, tsv
                )
                VALUES (
                    :document_id, :stable_id, 0,
                    0, :char_end, :text,
                    to_tsvector('english', :text)
                )
                RETURNING id
                """
            ),
            {
                "document_id": int(doc_id),
                "stable_id": stable_id,
                "char_end": len(chunk_text),
                "text": chunk_text,
            },
        ).scalar_one()

        # If embedding column exists, set a dummy vector so vector search doesn’t fail
        # (pgvector accepts a string like "[0,0,...]")
        s.execute(
            text(
                """
                UPDATE doc_chunks
                SET embedding = CAST(:vec AS vector)
                WHERE id = :id
                """
            ),
            {"vec": "[" + ",".join(["0.0"] * 384) + "]", "id": int(chunk_id)},
        )

        s.commit()

    return {"doc_id": int(doc_id), "chunk_id": int(chunk_id)}
