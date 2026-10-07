"""Business rules around the LLM call.

The LLM proposes an intent and an action; these rules decide what the customer actually gets.
Pure functions with no I/O and no conversation state, so every rule is unit-testable.
"""

from dataclasses import dataclass

from rag_engine.assistant import replies
from rag_engine.assistant.verify import body
from rag_engine.assistant.vocabulary import mentions_trevona, names_subject, retailers
from rag_engine.config import settings
from rag_engine.models import Action, Decision, Hit, Intent, Response, best_distance


@dataclass(frozen=True)
class Turn:
    """Everything the rules need to know about the message being answered."""

    message: str
    # The message plus earlier unresolved ones; what retrieval and answer checks run on.
    query: str
    hits: list[Hit]
    clarify_count: int
    user_messages: tuple[str, ...]

    @property
    def clarify_allowed(self) -> bool:
        return self.clarify_count < settings.max_clarify_turns

    @property
    def closest(self) -> float:
        return best_distance(self.hits)


def before_llm(turn: Turn) -> Response | None:
    """Decline clearly unrelated messages without calling the LLM, so it cannot be talked into
    helping. Skipped mid-clarification, where short follow-ups ("Amazon") look unrelated."""
    if turn.clarify_count or mentions_trevona(turn.message):
        return None
    if turn.closest >= settings.off_topic_min_distance:
        return _decline(turn)
    return None


def intent_of(turn: Turn, decision: Decision) -> Intent:
    """The LLM's intent, except an off-topic verdict too close to the docs to trust."""
    if decision.intent is Intent.OFF_TOPIC and turn.closest < settings.llm_off_topic_min_distance:
        return Intent.QUESTION
    return decision.intent


def asks_which_retailer(turn: Turn) -> bool:
    """A question that names no retailer, whose closest section answers it for one retailer while
    other close sections answer it for others ("how much is the fee?"). Any one answer is a guess.
    """
    if not turn.clarify_allowed or not turn.hits or names_subject(turn.query):
        return False
    closest = min(turn.hits, key=lambda h: h.distance)
    covered = {r for h in turn.hits for r in retailers(h.heading)}
    return bool(retailers(closest.heading)) and len(covered) > 1


def after_llm(turn: Turn, decision: Decision, documented: Hit | None) -> Response:
    """Turn the LLM's proposal into the reply the customer receives.

    `documented` is the section verified to answer the question (see verify.find_answer).
    """
    intent = intent_of(turn, decision)
    if intent is Intent.OFF_TOPIC:
        return _decline(turn)

    if intent is Intent.QUESTION and asks_which_retailer(turn):
        return Response(intent, Action.CLARIFY, replies.WHICH_RETAILER, None, turn.hits)

    # Only documented text is ever an answer, whatever action the model picked. Requests are
    # never answered: only staff can act on them.
    if intent is Intent.QUESTION and documented:
        return Response(intent, Action.ANSWER, body(documented), None, turn.hits, source=documented)

    if turn.clarify_allowed:
        # The model asked the customer something (sometimes mislabelled as a handoff).
        if decision.action in (Action.CLARIFY, Action.HANDOFF) and "?" in decision.reply:
            return Response(intent, Action.CLARIFY, decision.reply, None, turn.hits)
        if decision.action is Action.CLARIFY:
            return Response(intent, Action.CLARIFY, replies.CLARIFY_FALLBACK, None, turn.hits)

    # Engine-owned text: the model must not promise actions the bot cannot perform.
    reply = replies.REQUEST_HANDOFF if intent is Intent.REQUEST else replies.NO_ANSWER
    summary = decision.summary or " | ".join(turn.user_messages)
    return Response(intent, Action.HANDOFF, reply, summary, turn.hits)


def _decline(turn: Turn) -> Response:
    return Response(Intent.OFF_TOPIC, Action.DECLINE, replies.OFF_TOPIC, None, turn.hits)
