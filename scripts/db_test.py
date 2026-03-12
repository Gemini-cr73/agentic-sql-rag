from __future__ import annotations

import sys
from pathlib import Path

from sqlalchemy import text

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from app.db.session import new_session


def main() -> None:
    with new_session() as s:
        print("Testing DB connection...")
        result = s.execute(text("select 1")).scalar()
        print("select 1 =>", result)

        print("\nChecking documents schema...")
        rows = s.execute(
            text(
                """
                SELECT column_name, data_type
                FROM information_schema.columns
                WHERE table_name = 'documents'
                ORDER BY ordinal_position
                """
            )
        ).fetchall()
        print("documents columns:", rows)

        print("\nChecking doc_chunks schema...")
        rows = s.execute(
            text(
                """
                SELECT column_name, data_type
                FROM information_schema.columns
                WHERE table_name = 'doc_chunks'
                ORDER BY ordinal_position
                """
            )
        ).fetchall()
        print("doc_chunks columns:", rows)


if __name__ == "__main__":
    main()
