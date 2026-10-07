from pathlib import Path

import psycopg
from pgvector.psycopg import register_vector

from rag_engine.config import settings

SCHEMA_PATH = Path(__file__).parent / "schema.sql"


def init_schema(path: Path = SCHEMA_PATH) -> None:
    # The vector extension must exist before register_vector can look up its type.
    with psycopg.connect(settings.database_url, autocommit=True) as conn:
        conn.execute(path.read_text())


def connect() -> psycopg.Connection:
    conn = psycopg.connect(settings.database_url)
    register_vector(conn)
    return conn
