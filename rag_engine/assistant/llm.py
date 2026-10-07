"""The Ollama calls: decide on a reply, judge whether a section answers, summarise a handoff.

Every call runs at temperature 0. Output the model cuts off or garbles never raises: it reads as
"hand off" or "no answer".
"""

from enum import StrEnum

import ollama
from pydantic import BaseModel, ValidationError

from rag_engine.config import settings
from rag_engine.models import Action, Decision, Intent

_client = ollama.Client(host=settings.ollama_host)

# Used when the model's JSON is cut off or malformed. The rules still answer from the docs if a
# section verifiably answers the question.
_UNREADABLE = Decision(intent=Intent.QUESTION, action=Action.HANDOFF, reply="(unreadable)")


def _chat(
    system: str,
    messages: list[dict[str, str]],
    *,
    max_tokens: int,
    schema: type[BaseModel] | None = None,
) -> str:
    """One chat call; `schema` constrains the output to that model's JSON."""
    result = _client.chat(
        model=settings.ollama_model,
        messages=[{"role": "system", "content": system}, *messages],
        format=schema.model_json_schema() if schema else None,
        options={"temperature": 0, "num_predict": max_tokens},
        keep_alive="30m",
    )
    return result.message.content


# --- Decide: intent, action and reply -------------------------------------------------------


def decide(system: str, history: list[dict[str, str]]) -> Decision:
    """The model's proposal for the latest message in `history` (see prompt.py)."""
    content = _chat(system, history, max_tokens=300, schema=Decision)
    try:
        return Decision.model_validate_json(content)
    except ValidationError:
        return _UNREADABLE


# --- Judge: does this section answer the question? ------------------------------------------


class _Verdict(StrEnum):
    # Named outcomes, not a yes/no flag, which a small model reads as "is the answer yes?".
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
    "a different retailer or item than the one asked about.\n"
    '- For a "how much" question, the passage must give the amount or say when it is set; one '
    "that only explains how fees work does not answer it."
)


def quote_answer(question: str, passage: str) -> str | None:
    """The sentence of the passage that answers the question, or None."""
    content = _chat(
        _QUOTE_SYSTEM,
        [{"role": "user", "content": f"QUESTION:\n{question}\n\nPASSAGE:\n{passage}"}],
        max_tokens=250,  # room for a quoted list; output cut off mid-JSON reads as "no answer"
        schema=_Quote,
    )
    try:
        verdict = _Quote.model_validate_json(content)
    except ValidationError:
        return None
    return verdict.quote if verdict.verdict is _Verdict.ANSWERED else None


# --- Summarise a handoff for staff ----------------------------------------------------------

_SUMMARY_SYSTEM = (
    "You write ticket summaries for Trevona ACO support staff. The customer's messages below were "
    "not answered by the support bot. In one or two short sentences, say what the customer is "
    "asking or needs, with any details they gave (retailer, item, order, account). Write in the "
    'third person ("Customer asks..."). Do not answer it, and do not add anything the customer '
    "did not say. Reply with the summary only."
)


def summarize(messages: list[str]) -> str | None:
    """A short ticket summary of the customer's unanswered messages, or None if the model gave
    none."""
    numbered = "\n".join(f"{i}. {m}" for i, m in enumerate(messages, start=1))
    content = _chat(_SUMMARY_SYSTEM, [{"role": "user", "content": numbered}], max_tokens=100)
    return content.strip().strip('"').strip() or None
