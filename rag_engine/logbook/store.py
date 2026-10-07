"""Write each turn to Postgres: its session, its question_log row and its debug trace."""

from collections.abc import Callable
from pathlib import Path
from typing import Any

from psycopg.types.json import Jsonb

from rag_engine.config import settings
from rag_engine.knowledge import db
from rag_engine.models import Hit, Trace, TurnRecord

SCHEMA_PATH = Path(__file__).parent / "schema.sql"

# Stored with each trace so a reply can still be explained after these are retuned.
_TRACED_SETTINGS = {
    "ollama_model",
    "embed_model",
    "top_k",
    "verify_top_n",
    "max_clarify_turns",
    "llm_off_topic_min_distance",
    "off_topic_min_distance",
}

_SESSION_SQL = """
INSERT INTO sessions (id, channel) VALUES (%s, %s)
ON CONFLICT (id) DO UPDATE SET last_active_at = now()
"""

_END_SQL = "UPDATE sessions SET ended_at = now(), end_reason = %s WHERE id = %s"

_QUESTION_SQL = """
INSERT INTO question_log (session_id, turn, message, redacted, intent, action, reply, summary,
                          source_path, source_heading, composed, latency_ms)
VALUES (%(session_id)s, %(turn)s, %(message)s, %(redacted)s, %(intent)s, %(action)s, %(reply)s,
        %(summary)s, %(source_path)s, %(source_heading)s, %(composed)s, %(latency_ms)s)
RETURNING id
"""

_DEBUG_SQL = """
INSERT INTO question_debug (question_id, query, clarify_count, hits, gated, decision, checks,
                            timings_ms, settings)
VALUES (%(question_id)s, %(query)s, %(clarify_count)s, %(hits)s, %(gated)s, %(decision)s,
        %(checks)s, %(timings_ms)s, %(settings)s)
"""


def init_schema() -> None:
    db.init_schema(SCHEMA_PATH)


def _section(hit: Hit) -> dict[str, Any]:
    return {"path": hit.path, "heading": hit.heading}


def _question_row(r: TurnRecord) -> dict[str, Any]:
    response, source = r.response, r.response.source
    return {
        "session_id": r.session_id,
        "turn": r.turn,
        "message": r.message,
        "redacted": list(response.redacted),
        "intent": response.intent.value,
        "action": response.action.value,
        "reply": response.reply,
        "summary": response.summary,
        "source_path": source.path if source else None,
        "source_heading": source.heading if source else None,
        "composed": response.composed,
        "latency_ms": r.latency_ms,
    }


def _debug_row(question_id: int, r: TurnRecord, trace: Trace) -> dict[str, Any]:
    hits = [{**_section(h), "distance": h.distance} for h in r.response.hits]
    checks = [{**_section(c.hit), "quote": c.quote, "supported": c.supported} for c in trace.checks]
    decision = trace.decision.model_dump(mode="json") if trace.decision else None
    return {
        "question_id": question_id,
        "query": trace.query,
        "clarify_count": trace.clarify_count,
        "hits": Jsonb(hits),
        "gated": trace.gated,
        "decision": Jsonb(decision) if decision else None,
        "checks": Jsonb(checks),
        "timings_ms": Jsonb(trace.timings_ms),
        "settings": Jsonb(settings.model_dump(mode="json", include=_TRACED_SETTINGS)),
    }


def recorder(channel: str) -> Callable[[TurnRecord], None]:
    """An Engine recorder that logs every turn under this channel (cli, discord)."""

    def record(r: TurnRecord) -> None:
        if not settings.log_questions:
            return
        with db.connect() as conn:  # one transaction: commits on exit
            conn.execute(_SESSION_SQL, (r.session_id, channel))
            question_id = conn.execute(_QUESTION_SQL, _question_row(r)).fetchone()[0]
            if settings.log_debug and r.trace:
                conn.execute(_DEBUG_SQL, _debug_row(question_id, r, r.trace))
            if r.end_reason:
                conn.execute(_END_SQL, (r.end_reason, r.session_id))

    return record
