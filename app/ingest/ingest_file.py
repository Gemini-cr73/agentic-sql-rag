from __future__ import annotations

import argparse
import hashlib
import logging
from pathlib import Path
from typing import Any

from sqlalchemy import text

from app.db.session import new_session

log = logging.getLogger("app.ingest.ingest_file")

try:
    from app.ingest.chunking import chunk_text as _chunk_text  # type: ignore
except Exception:
    _chunk_text = None

ALLOWED_EXTENSIONS = {".txt", ".md"}


def _read_text_file(path: Path) -> str:
    return path.read_text(encoding="utf-8", errors="ignore")


def _stable_id_for_chunk(
    *,
    doc_id: int,
    chunk_index: int,
    char_start: int,
    char_end: int,
) -> str:
    raw = f"{doc_id}:{chunk_index}:{char_start}:{char_end}".encode()
    return hashlib.sha1(raw).hexdigest()


def _fallback_chunk_text(
    text_in: str,
    *,
    chunk_size: int = 800,
    overlap: int = 100,
) -> list[dict[str, Any]]:
    chunks: list[dict[str, Any]] = []

    if not text_in.strip():
        return chunks

    start = 0
    text_len = len(text_in)

    while start < text_len:
        end = min(start + chunk_size, text_len)
        chunk = text_in[start:end]

        if chunk.strip():
            chunks.append(
                {
                    "text": chunk,
                    "char_start": start,
                    "char_end": end,
                }
            )

        if end >= text_len:
            break

        start = max(end - overlap, start + 1)

    return chunks


def _chunk_document(
    text_in: str,
    *,
    chunk_size: int = 800,
    overlap: int = 100,
) -> list[dict[str, Any]]:
    if _chunk_text is not None:
        try:
            result = _chunk_text(text_in, chunk_size=chunk_size, overlap=overlap)
            normalized: list[dict[str, Any]] = []

            for item in result:
                if isinstance(item, dict):
                    normalized.append(
                        {
                            "text": item.get("text", ""),
                            "char_start": int(item.get("char_start", 0)),
                            "char_end": int(item.get("char_end", 0)),
                        }
                    )
                else:
                    normalized.append(
                        {
                            "text": str(item),
                            "char_start": 0,
                            "char_end": 0,
                        }
                    )
            return normalized
        except Exception:
            log.exception("Custom chunker failed; falling back to local chunker")

    return _fallback_chunk_text(text_in, chunk_size=chunk_size, overlap=overlap)


def ingest_text_file(
    file_path: str | Path,
    *,
    source: str = "upload",
    title: str | None = None,
    chunk_size: int = 800,
    overlap: int = 100,
) -> dict[str, Any]:
    path = Path(file_path)

    if not path.exists():
        raise FileNotFoundError(f"File not found: {path}")

    if path.suffix.lower() not in ALLOWED_EXTENSIONS:
        raise ValueError(
            f"Unsupported file type: {path.suffix}. Allowed: {sorted(ALLOWED_EXTENSIONS)}"
        )

    log.info("[ingest] starting file=%s", path)

    raw_text = _read_text_file(path)
    if not raw_text.strip():
        raise ValueError("File is empty after reading")

    doc_title = title or path.name
    chunks = _chunk_document(raw_text, chunk_size=chunk_size, overlap=overlap)

    if not chunks:
        raise ValueError("No chunks were produced from the uploaded file")

    log.info("[ingest] chunking complete chunks=%s", len(chunks))

    with new_session() as session:
        try:
            session.execute(text("SET LOCAL lock_timeout = '5s'"))
            session.execute(text("SET LOCAL statement_timeout = '15s'"))
            log.info("[ingest] db timeouts configured")

            log.info("[ingest] inserting document row")
            doc_insert = text(
                """
                INSERT INTO documents (source, title, content)
                VALUES (:source, :title, :content)
                RETURNING id
                """
            )

            doc_id = session.execute(
                doc_insert,
                {
                    "source": source,
                    "title": doc_title,
                    "content": raw_text,
                },
            ).scalar_one()

            log.info("[ingest] document inserted doc_id=%s", doc_id)

            chunk_insert = text(
                """
                INSERT INTO doc_chunks
                    (document_id, chunk_index, text, stable_id, char_start, char_end)
                VALUES
                    (:document_id, :chunk_index, :text, :stable_id, :char_start, :char_end)
                """
            )

            inserted = 0
            for idx, chunk in enumerate(chunks):
                chunk_text = (chunk.get("text") or "").strip()
                if not chunk_text:
                    continue

                char_start = int(chunk.get("char_start", 0))
                char_end = int(chunk.get("char_end", 0))

                stable_id = _stable_id_for_chunk(
                    doc_id=doc_id,
                    chunk_index=idx,
                    char_start=char_start,
                    char_end=char_end,
                )

                session.execute(
                    chunk_insert,
                    {
                        "document_id": doc_id,
                        "chunk_index": idx,
                        "text": chunk_text,
                        "stable_id": stable_id,
                        "char_start": char_start,
                        "char_end": char_end,
                    },
                )
                inserted += 1

            log.info("[ingest] all chunk inserts queued inserted=%s", inserted)

            session.commit()
            log.info("[ingest] commit complete doc_id=%s inserted=%s", doc_id, inserted)

            return {
                "document_id": doc_id,
                "title": doc_title,
                "source": source,
                "chunks_inserted": inserted,
            }

        except Exception:
            log.exception("[ingest] failure; rolling back")
            session.rollback()
            raise


def main() -> None:
    parser = argparse.ArgumentParser(description="Ingest a .txt or .md file")
    parser.add_argument("file_path", help="Path to .txt or .md file")
    parser.add_argument("--source", default="cli", help="Document source label")
    parser.add_argument("--title", default=None, help="Optional title override")
    parser.add_argument("--chunk-size", type=int, default=800)
    parser.add_argument("--overlap", type=int, default=100)
    args = parser.parse_args()

    result = ingest_text_file(
        args.file_path,
        source=args.source,
        title=args.title,
        chunk_size=args.chunk_size,
        overlap=args.overlap,
    )
    print(result)


if __name__ == "__main__":
    main()
