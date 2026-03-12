from __future__ import annotations

from pydantic import BaseModel


class DocumentIngestResponse(BaseModel):
    document_id: int
    title: str
    source: str
    chunks_inserted: int
    embeddings_backfilled: int
    message: str


class DocumentItem(BaseModel):
    id: int
    title: str | None = None
    source: str
    created_at: str | None = None
    chunk_count: int = 0


class DocumentListResponse(BaseModel):
    documents: list[DocumentItem]
    total: int


class DocumentDeleteResponse(BaseModel):
    document_id: int
    deleted: bool
    message: str
