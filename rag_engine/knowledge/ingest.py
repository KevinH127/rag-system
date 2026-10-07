import hashlib
from dataclasses import dataclass
from pathlib import Path

from rag_engine.config import settings
from rag_engine.knowledge import db
from rag_engine.knowledge.chunking import chunk_markdown
from rag_engine.knowledge.embed import embed_documents


@dataclass
class IngestReport:
    added: int = 0
    updated: int = 0
    unchanged: int = 0
    removed: int = 0
    chunks: int = 0


def ingest(root: Path | None = None) -> IngestReport:
    """Sync the knowledge base directory into Postgres, skipping unchanged files."""
    root = root or settings.knowledge_base_dir
    files = {str(p.relative_to(root)): p for p in sorted(root.rglob("*.md"))}
    report = IngestReport()

    with db.connect() as conn:
        known = dict(conn.execute("SELECT path, content_hash FROM documents").fetchall())

        for path in known.keys() - files.keys():
            conn.execute("DELETE FROM documents WHERE path = %s", (path,))
            report.removed += 1

        for path, file in files.items():
            text = file.read_text()
            digest = hashlib.sha256(text.encode()).hexdigest()
            if known.get(path) == digest:
                report.unchanged += 1
                continue

            chunks = chunk_markdown(text)
            vectors = embed_documents([c.content for c in chunks])

            conn.execute("DELETE FROM documents WHERE path = %s", (path,))
            doc_id = conn.execute(
                "INSERT INTO documents (path, content_hash) VALUES (%s, %s) RETURNING id",
                (path, digest),
            ).fetchone()[0]
            with conn.cursor() as cur:
                cur.executemany(
                    "INSERT INTO chunks (document_id, ord, heading, content, embedding)"
                    " VALUES (%s, %s, %s, %s, %s)",
                    [
                        (doc_id, c.ord, c.heading, c.content, v)
                        for c, v in zip(chunks, vectors, strict=True)
                    ],
                )
            report.chunks += len(chunks)
            if path in known:
                report.updated += 1
            else:
                report.added += 1
        conn.commit()
    return report
