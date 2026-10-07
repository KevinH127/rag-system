"""Types shared by every layer."""

from dataclasses import dataclass, field
from enum import StrEnum
from uuid import UUID

from pydantic import BaseModel, Field


class Intent(StrEnum):
    QUESTION = "question"
    REQUEST = "request"
    OFF_TOPIC = "off_topic"


class Action(StrEnum):
    ANSWER = "answer"
    CLARIFY = "clarify"
    HANDOFF = "handoff"
    DECLINE = "decline"


class Decision(BaseModel):
    """Structured output the LLM must produce (used as the Ollama JSON schema)."""

    intent: Intent
    action: Action
    reply: str = Field(min_length=1)
    summary: str | None = None


@dataclass(frozen=True)
class Hit:
    """One retrieved knowledge-base section."""

    path: str
    heading: str
    content: str
    distance: float


def best_distance(hits: list[Hit]) -> float:
    """Closest vector distance among the hits (fusion can put a closer chunk below rank 1)."""
    return min((h.distance for h in hits), default=1.0)


@dataclass(frozen=True)
class Response:
    """What the customer receives, plus what staff and debugging need."""

    intent: Intent
    action: Action
    reply: str
    summary: str | None = None
    hits: list[Hit] = field(default_factory=list)
    redacted: tuple[str, ...] = ()
    # The section an answer was taken from, verbatim.
    source: Hit | None = None


@dataclass(frozen=True)
class Check:
    """One judge call: the section shown, the sentence quoted from it, and whether it held up."""

    hit: Hit
    quote: str | None
    supported: bool


@dataclass(frozen=True)
class Trace:
    """How a reply was reached. Kept for debugging, never shown to the customer."""

    # The message plus earlier unresolved ones; what retrieval and the judge ran on.
    query: str
    clarify_count: int
    # Declined by the rules before the LLM was called.
    gated: bool
    # The model's own proposal, before the rules decided the reply.
    decision: Decision | None = None
    checks: tuple[Check, ...] = ()
    timings_ms: dict[str, int] = field(default_factory=dict)


@dataclass(frozen=True)
class TurnRecord:
    """One customer message and its reply, as handed to the engine's recorder."""

    session_id: UUID
    turn: int
    # Redacted text; the raw message is never recorded.
    message: str
    response: Response
    # None when nothing ran: the whole message was a secret, or the session was already closed.
    trace: Trace | None
    latency_ms: int
    # Set on the turn that ended the session ("handoff").
    end_reason: str | None = None
