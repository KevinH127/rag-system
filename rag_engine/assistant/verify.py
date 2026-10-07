"""Which doc section, if any, answers the customer's question.

A small model invents plausible answers instead of saying it does not know, and neither embedding
distance nor word overlap can tell "answered by the docs" from "merely related to the docs". So a
judge (llm.quote_answer) must quote the sentence that answers the question, and the quote is
checked here against the section text. Only a question with a verified section is answered;
grounding.py then decides whether the model's own wording of the answer can be used.
"""

import re
from collections.abc import Callable

from rag_engine.assistant.vocabulary import retailers
from rag_engine.config import settings
from rag_engine.models import Check, Hit

# (question, section content) -> the sentence that answers the question, or None.
Judge = Callable[[str, str], str | None]

# A quote must be at least this long, and its first _MATCH_CHARS characters must be in the text.
MIN_QUOTE_CHARS = 10
_MATCH_CHARS = 40


def check_sections(
    question: str, hits: list[Hit], judge: Judge, *, general: bool = False
) -> list[Check]:
    """Judge the closest few sections in order, stopping at the first verified answer.

    Sections about a retailer the question does not ask about are never judged. For a `general`
    answer (one covering every retailer), sections whose heading names no retailer go first. The
    quote is checked against the body only, so quoting the heading back does not count.
    """

    def order(hit: Hit) -> tuple[bool, float]:
        return general and bool(retailers(hit.heading)), hit.distance

    candidates = sorted((h for h in hits if not about_another_retailer(h, question)), key=order)
    checks: list[Check] = []
    for hit in candidates[: settings.verify_top_n]:
        quote = judge(question, hit.content)
        checks.append(Check(hit, quote, bool(quote) and is_supported(quote, body(hit))))
        if checks[-1].supported:
            break
    return checks


def verified(checks: list[Check]) -> Check | None:
    """The check that passed, if any."""
    return next((c for c in checks if c.supported), None)


def about_another_retailer(hit: Hit, question: str) -> bool:
    """True if the question names retailers and the section covers only others: its heading
    names only others or, with no retailer in its heading, its text does. Stops the judge
    accepting one retailer's answer (or an overview of the others) for another retailer."""
    asked = retailers(question)
    named = retailers(hit.heading) or retailers(body(hit))
    return bool(asked and named and not asked & named)


def is_supported(quote: str, text: str) -> bool:
    """True if the quote really appears in the text (ignoring case, punctuation and spacing)."""
    q = _normalise(quote)
    return len(q) >= MIN_QUOTE_CHARS and q[:_MATCH_CHARS] in _normalise(text)


def body(hit: Hit) -> str:
    """Section text without its "Title > heading" label line."""
    _, _, text = hit.content.partition("\n\n")
    return text.strip()


def verbatim(hit: Hit) -> str:
    """The section as sent to a customer: its text without a bare opening "Yes." or "No.".

    That word answers the section's heading, which the customer never sees and may have asked
    the opposite way ("Can I get a refund?" against "Do I still pay the fee?").
    """
    return re.sub(r"^(?:Yes|No)\.\s+", "", body(hit))


def _normalise(text: str) -> str:
    return re.sub(r"[^a-z0-9$]+", " ", text.lower()).strip()
