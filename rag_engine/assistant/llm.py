from enum import StrEnum

import ollama
from pydantic import BaseModel, ValidationError

from rag_engine.config import settings
from rag_engine.models import Action, Decision, Intent

_client = ollama.Client(host=settings.ollama_host)

# Used when the model's JSON is cut off or malformed: hand off rather than crash. The rules then
# still answer from the docs if a section verifiably answers the question.
_UNREADABLE = Decision(intent=Intent.QUESTION, action=Action.HANDOFF, reply="(unreadable)")


def decide(system: str, history: list[dict[str, str]]) -> Decision:
    """One structured chat call returning intent, action and reply."""
    result = _client.chat(
        model=settings.ollama_model,
        messages=[{"role": "system", "content": system}, *history],
        format=Decision.model_json_schema(),
        options={"temperature": 0, "num_predict": 300},
        keep_alive="30m",
    )
    try:
        return Decision.model_validate_json(result.message.content)
    except ValidationError:
        return _UNREADABLE


class _Verdict(StrEnum):
    # Named outcomes, not a yes/no: given `answers: bool`, a 3B model marked correct "No, that is
    # not accepted" answers as false, reading the flag as "is the answer yes?".
    ANSWERED = "passage_answers_question"
    NOT_ANSWERED = "answer_not_in_passage"


class _Quote(BaseModel):
    # Quote first: writing out the evidence before the verdict makes a small model more accurate.
    quote: str
    verdict: _Verdict


_QUOTE_SYSTEM = (
    "A customer asked a question. Decide whether the passage contains the answer.\n"
    '- Copy the sentence from the passage that answers the question into "quote" (empty if none).\n'
    '- Set "verdict" to "passage_answers_question" if the passage answers the question. A "no", '
    '"not accepted", "closed" or "set later" answer still counts as answering it.\n'
    '- Set "verdict" to "answer_not_in_passage" if the passage is about something else, or about '
    "a different retailer or item than the one asked about."
)


def quote_answer(question: str, passage: str) -> str | None:
    """The sentence of the passage that answers the question, or None."""
    result = _client.chat(
        model=settings.ollama_model,
        messages=[
            {"role": "system", "content": _QUOTE_SYSTEM},
            {"role": "user", "content": f"QUESTION:\n{question}\n\nPASSAGE:\n{passage}"},
        ],
        format=_Quote.model_json_schema(),
        # Room for a long quoted list; output cut off mid-JSON is treated as "no answer".
        options={"temperature": 0, "num_predict": 250},
        keep_alive="30m",
    )
    try:
        verdict = _Quote.model_validate_json(result.message.content)
    except ValidationError:
        return None
    return verdict.quote if verdict.verdict is _Verdict.ANSWERED else None
