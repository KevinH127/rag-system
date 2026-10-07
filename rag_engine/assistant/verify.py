"""Only documented text is ever given as an answer.

A 3B model invents plausible answers ("Yes, your card is encrypted") instead of saying it does
not know, and neither embedding distance nor word overlap can tell "answered by the docs" from
"merely related to the docs" (measured on evals/golden.jsonl). So a judge must quote the sentence
that answers the question, the quote is checked here against the section text, and the first
section that passes becomes the answer, verbatim.
"""

import re
from collections.abc import Callable

from rag_engine.config import settings
from rag_engine.models import Check, Hit

# (question, section content) -> the sentence that answers the question, or None.
Judge = Callable[[str, str], str | None]

MIN_QUOTE_CHARS = 10
_MATCH_CHARS = 40


def body(hit: Hit) -> str:
    """Section text without its "Title > heading" label line."""
    _, _, text = hit.content.partition("\n\n")
    return text.strip()


def _normalise(text: str) -> str:
    return re.sub(r"[^a-z0-9$]+", " ", text.lower()).strip()


def is_supported(quote: str, text: str) -> bool:
    """True if the quote really appears in the text (ignoring case, punctuation and spacing)."""
    q = _normalise(quote)
    return len(q) >= MIN_QUOTE_CHARS and q[:_MATCH_CHARS] in _normalise(text)


def check_sections(question: str, hits: list[Hit], judge: Judge) -> list[Check]:
    """Judge the closest few sections in order, stopping at the first verified answer.

    The quote is checked against the body only, so quoting the heading back does not count.
    """
    checks: list[Check] = []
    for hit in sorted(hits, key=lambda h: h.distance)[: settings.verify_top_n]:
        quote = judge(question, hit.content)
        checks.append(Check(hit, quote, bool(quote) and is_supported(quote, body(hit))))
        if checks[-1].supported:
            break
    return checks


def answer_of(checks: list[Check]) -> Hit | None:
    """The section that passed, if any."""
    return next((c.hit for c in checks if c.supported), None)


def find_answer(question: str, hits: list[Hit], judge: Judge) -> Hit | None:
    """The closest section the judge can quote an answer from, checking the top few in order."""
    return answer_of(check_sections(question, hits, judge))
