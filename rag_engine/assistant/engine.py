import logging
import time
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from dataclasses import dataclass, field, replace
from uuid import UUID, uuid4

from rag_engine.assistant import llm, policy, replies
from rag_engine.assistant.prompt import build_system
from rag_engine.assistant.redact import drop_secret_clauses, redact
from rag_engine.assistant.verify import Judge, answer_of, check_sections
from rag_engine.assistant.vocabulary import only_names_subject, without_subjects
from rag_engine.knowledge import retrieve
from rag_engine.models import Action, Check, Decision, Hit, Intent, Response, Trace, TurnRecord

log = logging.getLogger(__name__)

Retriever = Callable[[str], list[Hit]]
Decider = Callable[[str, list[dict[str, str]]], Decision]
Recorder = Callable[[TurnRecord], None]


def _no_record(record: TurnRecord) -> None:
    pass


def _ms_since(start: float) -> int:
    return round((time.perf_counter() - start) * 1000)


@contextmanager
def _timed(timings: dict[str, int], stage: str) -> Iterator[None]:
    start = time.perf_counter()
    try:
        yield
    finally:
        timings[stage] = _ms_since(start)


@dataclass
class Engine:
    """One customer conversation (session): its state, and the pipeline each message goes through.

    redact -> retrieve -> rules before the LLM -> LLM -> verify an answer in the docs
    -> rules after the LLM -> update state -> record
    """

    retriever: Retriever = retrieve.search
    decider: Decider = llm.decide
    judge: Judge = llm.quote_answer
    # Receives every turn; interfaces pass a log store, tests and evals keep the no-op.
    recorder: Recorder = _no_record
    session_id: UUID = field(default_factory=uuid4)
    history: list[dict[str, str]] = field(default_factory=list, init=False)
    clarify_count: int = field(default=0, init=False)
    # Set by a handoff: staff take over in a ticket, so this conversation is over.
    closed: bool = field(default=False, init=False)
    # User messages since the last resolved turn; retrieved together so follow-ups keep context.
    _pending: list[str] = field(default_factory=list, init=False, repr=False)
    # The last resolved question without its retailer, re-asked by a bare "what about Costco?".
    _topic: str = field(default="", init=False, repr=False)
    _turn: int = field(default=0, init=False, repr=False)

    def respond(self, message: str) -> Response:
        start = time.perf_counter()
        was_closed = self.closed
        cleaned = redact(message)
        text = drop_secret_clauses(cleaned.text) if cleaned.kinds else cleaned.text
        if was_closed:
            response, trace = Response(Intent.QUESTION, Action.HANDOFF, replies.CLOSED), None
        elif text:
            response, trace = self._respond(text)
        else:
            response, trace = Response(Intent.QUESTION, Action.CLARIFY, replies.ASK_AGAIN), None
        if cleaned.kinds:
            warned = f"{replies.SECRET_WARNING}\n\n{response.reply}"
            response = replace(response, reply=warned, redacted=cleaned.kinds)
        self._turn += 1
        end_reason = "handoff" if self.closed and not was_closed else None
        self._record(
            TurnRecord(
                self.session_id,
                self._turn,
                cleaned.text,
                response,
                trace,
                _ms_since(start),
                end_reason,
            )
        )
        return response

    def _respond(self, message: str) -> tuple[Response, Trace]:
        if not self._pending and self._topic and only_names_subject(message):
            self._pending.append(self._topic)
        self._pending.append(message)
        self.history.append({"role": "user", "content": message})
        query = " ".join(self._pending)
        timings: dict[str, int] = {}
        with _timed(timings, "retrieve"):
            hits = self.retriever(query)
        turn = policy.Turn(
            message=message,
            query=query,
            hits=hits,
            clarify_count=self.clarify_count,
            user_messages=tuple(m["content"] for m in self.history if m["role"] == "user"),
        )
        decision, checks = None, ()
        response = policy.before_llm(turn)
        gated = response is not None
        if response is None:
            response, decision, checks = self._decide(turn, timings)
        self._remember(response)
        return response, Trace(query, turn.clarify_count, gated, decision, checks, timings)

    def _decide(
        self, turn: policy.Turn, timings: dict[str, int]
    ) -> tuple[Response, Decision, tuple[Check, ...]]:
        with _timed(timings, "decide"):
            decision = self.decider(build_system(turn.hits, turn.clarify_allowed), self.history)
        checks: list[Check] = []
        intent = policy.intent_of(turn, decision)
        # No answer is given while the retailer is unknown, so there is nothing to verify yet.
        if intent is Intent.QUESTION and not policy.asks_which_retailer(turn):
            with _timed(timings, "verify"):
                checks = check_sections(turn.query, turn.hits, self.judge)
        response = policy.after_llm(turn, decision, answer_of(checks))
        return response, decision, tuple(checks)

    def _remember(self, response: Response) -> None:
        self.history.append({"role": "assistant", "content": response.reply})
        if response.action is Action.DECLINE:
            self._pending.pop()  # an off-topic message must not pollute the next retrieval
        elif response.action is Action.CLARIFY:
            self.clarify_count += 1
        else:
            self.clarify_count = 0
            self.closed = response.action is Action.HANDOFF
            asked = [m for m in self._pending if not only_names_subject(m)]
            self._topic = without_subjects(" ".join(asked))
            self._pending.clear()

    def _record(self, record: TurnRecord) -> None:
        try:
            self.recorder(record)
        except Exception:
            # A logging failure must never cost the customer their reply.
            log.exception("Could not record turn %d of session %s", record.turn, record.session_id)
