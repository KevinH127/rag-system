"""Business rules around the LLM call.

The LLM proposes an intent and an action; these rules decide what the customer actually gets.
Pure functions with no I/O and no conversation state, so every rule is unit-testable.
"""

from dataclasses import dataclass

from rag_engine.assistant import replies
from rag_engine.assistant.verify import grounded, verbatim
from rag_engine.assistant.vocabulary import (
    asks_in_general,
    mentions_trevona,
    names_subject,
    retailers,
    small_talk,
)
from rag_engine.config import settings
from rag_engine.models import Action, Check, Decision, Hit, Intent, Response, best_distance


@dataclass(frozen=True)
class Turn:
    """Everything the rules need to know about the message being answered."""

    message: str
    # The message plus earlier unresolved ones; what retrieval and answer checks run on.
    query: str
    hits: list[Hit]
    clarify_count: int
    # The message only re-asks the last answered question for another retailer or in general.
    follows_up: bool = False

    @property
    def clarify_allowed(self) -> bool:
        return self.clarify_count < settings.max_clarify_turns

    @property
    def closest(self) -> float:
        return best_distance(self.hits)


def before_llm(turn: Turn) -> Response | None:
    """Reply to pleasantries, and decline clearly unrelated messages, without calling the LLM, so
    it cannot be talked into helping. The distance gate is skipped mid-clarification, where short
    follow-ups ("Amazon") look unrelated."""
    if small_talk(turn.message):
        return _decline(turn)
    if turn.clarify_count or mentions_trevona(turn.message):
        return None
    if turn.closest >= settings.off_topic_min_distance:
        return _decline(turn)
    return None


def intent_of(turn: Turn, decision: Decision) -> Intent:
    """The LLM's intent, except an off-topic verdict too close to the docs to trust, and a bare
    follow-up, which continues the question before it (measured: "what about costco?" after a
    Walmart fee answer was labelled a request and handed to staff)."""
    if turn.follows_up:
        return Intent.QUESTION
    if decision.intent is Intent.OFF_TOPIC and turn.closest < settings.llm_off_topic_min_distance:
        return Intent.QUESTION
    return decision.intent


def varies_by_retailer(turn: Turn) -> bool:
    """A question naming no retailer or product, whose closest section answers it for one
    retailer while other close sections answer it for others ("how much is the fee?")."""
    if not turn.hits or names_subject(turn.query):
        return False
    closest = min(turn.hits, key=lambda h: h.distance)
    covered = {r for h in turn.hits for r in retailers(h.heading)}
    return bool(retailers(closest.heading)) and len(covered) > 1


def wants_general_answer(turn: Turn) -> bool:
    """The answer should cover every retailer, not the closest one: the customer said so ("in
    general", "for all of them"), or named none while the answer varies by retailer. The bot
    summarises rather than asking which retailer. Measured: the Walmart fee section was closer to
    "how much is the aco fee in general" than the general overview was."""
    if retailers(turn.query):
        return False
    return asks_in_general(turn.message) or varies_by_retailer(turn)


def prompt_sections(turn: Turn, general: bool) -> list[Hit]:
    """The sections shown to the model, sections tied to no retailer first for a general answer
    (the prompt tells it to answer from the earliest section)."""
    if not general:
        return turn.hits
    return sorted(turn.hits, key=lambda h: bool(retailers(h.heading)))


def asks_back(turn: Turn, decision: Decision) -> bool:
    """The model asks the customer about a question too vague to look up ("it's not working").
    Judged against such a message, the judge verified an unrelated section ("A decline can show
    up twice in the team's reports"). The only question the bot ever asks: never for a request,
    a bare follow-up, or an answer that varies by retailer (that is summarised instead)."""
    return (
        decision.action is Action.CLARIFY
        and "?" in decision.reply
        and turn.clarify_allowed
        and intent_of(turn, decision) is Intent.QUESTION
        and not turn.follows_up
        and not wants_general_answer(turn)
    )


def after_llm(turn: Turn, decision: Decision, answer: Check | None) -> Response:
    """Turn the LLM's proposal into the reply the customer receives.

    `answer` is the judge's check that verified a section answers the question (see
    verify.check_sections).
    """
    intent = intent_of(turn, decision)
    if intent is Intent.OFF_TOPIC:
        return _decline(turn)

    if asks_back(turn, decision):
        return Response(intent, Action.CLARIFY, decision.reply, None, turn.hits)

    # Only a question the docs verifiably answer is answered, whatever else the model picked.
    # Requests are never answered: only staff can act on them.
    if intent is Intent.QUESTION and answer:
        return _answer(turn, decision, answer)

    # A request, or a question the docs do not answer, goes straight to staff.
    # Engine-owned text: the model must not promise actions the bot cannot perform. The engine
    # adds the ticket summary of what was not answered.
    reply = replies.REQUEST_HANDOFF if intent is Intent.REQUEST else replies.NO_ANSWER
    return Response(intent, Action.HANDOFF, reply, None, turn.hits)


def _answer(turn: Turn, decision: Decision, answer: Check) -> Response:
    """The model's own wording if it holds up against the docs, else the section verbatim."""
    composed = decision.action is Action.ANSWER and grounded(
        decision.reply, answer, turn.hits, turn.query
    )
    reply = decision.reply if composed else verbatim(answer.hit)
    return Response(
        Intent.QUESTION,
        Action.ANSWER,
        reply,
        None,
        turn.hits,
        source=answer.hit,
        composed=composed,
    )


def _decline(turn: Turn) -> Response:
    """Off-topic, or small talk, which gets a friendly reply instead."""
    kind = small_talk(turn.message)
    reply = replies.SMALL_TALK[kind] if kind else replies.OFF_TOPIC
    return Response(Intent.OFF_TOPIC, Action.DECLINE, reply, None, turn.hits)
