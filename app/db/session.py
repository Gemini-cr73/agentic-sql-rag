from __future__ import annotations

import os

from sqlalchemy import create_engine
from sqlalchemy.orm import Session, sessionmaker


def _normalize_database_url(url: str) -> str:
    """
    Normalize provider URLs for SQLAlchemy.

    Railway/managed Postgres URLs may come in different forms:
    - postgres://...
    - postgresql://...
    - postgresql+psycopg://...

    This project uses SQLAlchemy + psycopg, so we normalize to:
    postgresql+psycopg://...
    """
    if url.startswith("postgresql+psycopg://"):
        return url
    if url.startswith("postgresql://"):
        return url.replace("postgresql://", "postgresql+psycopg://", 1)
    if url.startswith("postgres://"):
        return url.replace("postgres://", "postgresql+psycopg://", 1)
    return url


DATABASE_URL = os.getenv("DATABASE_URL")
if not DATABASE_URL:
    raise RuntimeError(
        "DATABASE_URL is not set. Set it in Railway variables or load it locally from an env file."
    )

DATABASE_URL = _normalize_database_url(DATABASE_URL)

engine = create_engine(
    DATABASE_URL,
    future=True,
    pool_pre_ping=True,
)

SessionLocal = sessionmaker(
    bind=engine,
    autoflush=False,
    autocommit=False,
    expire_on_commit=False,
    class_=Session,
)


def new_session() -> Session:
    return SessionLocal()
