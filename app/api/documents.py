from __future__ import annotations

import logging
import os
import tempfile
from pathlib import Path

from fastapi import APIRouter, File, HTTPException, UploadFile
from sqlalchemy import select

from app.db.models import Document
from app.db.session import new_session
from app.ingest.ingest_file import ingest_text_file

router = APIRouter(prefix="/documents", tags=["documents"])
log = logging.getLogger("app.api.documents")

RUN_EMBEDDINGS_ON_UPLOAD = False


@router.get("")
@router.get("/")
def list_documents():
    with new_session() as session:
        rows = (
            session.execute(select(Document).order_by(Document.id.desc()).limit(50))
            .scalars()
            .all()
        )

        return {
            "documents": [
                {
                    "id": row.id,
                    "source": row.source,
                    "title": row.title,
                }
                for row in rows
            ],
            "total": len(rows),
        }


@router.post("/upload")
async def upload_document(file: UploadFile = File(...)):
    temp_path: Path | None = None

    try:
        log.info("[documents.upload] starting upload filename=%s", file.filename)

        if not file.filename:
            raise HTTPException(status_code=400, detail="Missing filename")

        suffix = Path(file.filename).suffix.lower()
        if suffix not in {".txt", ".md"}:
            raise HTTPException(
                status_code=400,
                detail="Only .txt and .md files are allowed",
            )

        data = await file.read()
        log.info("[documents.upload] file read complete bytes=%s", len(data))

        if not data:
            raise HTTPException(status_code=400, detail="Uploaded file is empty")

        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
            tmp.write(data)
            temp_path = Path(tmp.name)

        log.info("[documents.upload] temp file created path=%s", temp_path)

        result = ingest_text_file(
            temp_path,
            source="streamlit_upload",
            title=file.filename,
            chunk_size=800,
            overlap=100,
        )

        log.info("[documents.upload] ingest completed result=%s", result)

        return {
            "status": "ok",
            "document": result,
            "embeddings_started": RUN_EMBEDDINGS_ON_UPLOAD,
        }

    except HTTPException:
        raise
    except Exception as exc:
        log.exception("[documents.upload] upload failed")
        raise HTTPException(status_code=500, detail=f"Upload failed: {exc}") from exc
    finally:
        try:
            await file.close()
        except Exception:
            pass

        if temp_path and temp_path.exists():
            try:
                os.remove(temp_path)
                log.info("[documents.upload] temp file removed path=%s", temp_path)
            except Exception:
                log.exception("[documents.upload] failed to remove temp file")
