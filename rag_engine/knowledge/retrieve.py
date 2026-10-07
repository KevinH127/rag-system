import re

from rag_engine.config import settings
from rag_engine.knowledge import db
from rag_engine.knowledge.embed import embed_query
from rag_engine.models import Hit

CANDIDATES = 20
RRF_K = 60

_VECTOR_SQL = """
SELECT c.id, d.path, c.heading, c.content, c.embedding <=> %(vec)s::vector AS distance
FROM chunks c JOIN documents d ON d.id = c.document_id
ORDER BY distance
LIMIT %(n)s
"""

# Keyword candidates (any query term may match), still scored by vector distance for the gates.
_KEYWORD_SQL = """
SELECT c.id, d.path, c.heading, c.content, c.embedding <=> %(vec)s::vector AS distance
FROM chunks c JOIN documents d ON d.id = c.document_id
WHERE c.tsv @@ to_tsquery('english', %(q)s)
ORDER BY ts_rank_cd(c.tsv, to_tsquery('english', %(q)s)) DESC
LIMIT %(n)s
"""


def _or_query(text: str) -> str:
    words = dict.fromkeys(re.findall(r"[a-z0-9]+", text.lower()))
    return " | ".join(words)


def search(query: str, k: int | None = None) -> list[Hit]:
    """Hybrid search: vector and keyword rankings fused with reciprocal rank fusion."""
    k = k or settings.top_k
    vec = embed_query(query)
    tsq = _or_query(query)
    with db.connect() as conn:
        by_vector = conn.execute(_VECTOR_SQL, {"vec": vec, "n": CANDIDATES}).fetchall()
        by_keyword = (
            conn.execute(_KEYWORD_SQL, {"vec": vec, "q": tsq, "n": CANDIDATES}).fetchall()
            if tsq
            else []
        )

    scores: dict[int, float] = {}
    rows: dict[int, tuple] = {}
    for ranking in (by_vector, by_keyword):
        for rank, row in enumerate(ranking):
            scores[row[0]] = scores.get(row[0], 0.0) + 1.0 / (RRF_K + rank + 1)
            rows[row[0]] = row
    best = sorted(scores, key=scores.__getitem__, reverse=True)[:k]
    # Fusion decides WHICH chunks to include; the LLM sees them closest-meaning first.
    return sorted((Hit(*rows[i][1:]) for i in best), key=lambda h: h.distance)
