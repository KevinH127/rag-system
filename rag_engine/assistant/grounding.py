"""Whether the model's own wording of an answer says only what the docs say.

Only questions with a verified section are answered at all (verify.py). The reply may then be the
model's own wording, but only if `grounded` finds nothing in it the docs do not say:

1. every amount, number, link, email and channel is in the sections;
2. each fee is said for the retailer and product the docs give it for;
3. it says yes or no the same way as the judge's quote;
4. it adds no claim of its own.

Otherwise the verified section is sent as written (verify.verbatim). The thresholds were tuned
on the golden set; see "Decision log" in docs/architecture.md.
"""

import re
from collections.abc import Iterator

from rag_engine.assistant.verify import body
from rag_engine.assistant.vocabulary import products, retailers, terms
from rag_engine.models import Check, Hit

# Words a faithful rewording adds without adding a claim ("Yes, you can also follow the guide").
_CONVERSATIONAL = frozenset(
    terms(
        """
        yes no also information info more follow find need check go click help sure note please
        just may can will then step instruction guide re ll ve don doesn isn aren didn won able
        let know like make ensure simply first once here there what when which how anything else
        thing way contact question ask asked answer
        """
    )
)
# Words a reply may use that appear in no section. Invented claims add more.
MAX_NEW_WORDS = 2
# Share of a reply sentence's words that one doc sentence (or heading) must contain.
MIN_SENTENCE_SUPPORT = 0.34
# Shared words needed before a reply clause and a quote clause are compared for yes or no;
# fewer pair unrelated clauses.
_MIN_CLAUSE_OVERLAP = 3

_AMOUNT = re.compile(
    r"\$\s?(\d[\d,]*(?:\.\d+)?)|(\d[\d,]*(?:\.\d+)?)\s?(?:CAD|dollars)\b", re.IGNORECASE
)
# Links, emails, domains and #channels must appear in the docs exactly.
_LITERAL = re.compile(r"https?://\S+|[\w.+-]+@[\w-]+\.[\w.]+|#[\w-]+|\b[\w-]+\.(?:ca|com)\b\S*")
_NUMBER = re.compile(r"\d+(?:\.\d+)?")
_NEGATION = re.compile(
    r"\b(?:no|not|never|cannot|nothing|none|nope)\b|n't\b|\bcant\b", re.IGNORECASE
)
_OPENER = re.compile(
    r"^\W*(?:(yes|yeah|yep|sure|absolutely)|no|nope|not|unfortunately)\b", re.IGNORECASE
)
_CLAUSE = re.compile(r"[,;(]|\b(?:and|while|but)\b")

# (amount, retailers, products) of one fee as a text states it.
FeeMention = tuple[str, set[str], set[str]]


def grounded(reply: str, answer: Check, hits: list[Hit], question: str) -> bool:
    """True if the model's own answer says nothing the docs do not (see the module docstring).

    `answer` is the judge's verified check, `hits` the sections the model saw, and `question`
    the customer's question as retrieved (with earlier unresolved messages).
    """
    if not reply.strip() or not answer.quote:
        return False
    docs = "\n".join(body(h) for h in hits)
    return (
        _facts_in_docs(reply, docs)
        and _fees_tied_to_what_the_docs_tie_them_to(reply, hits, question)
        and _agrees_with_quote(reply, answer.quote)
        and _says_nothing_new(reply, hits, question)
    )


# --- 1. Amounts, numbers, links ------------------------------------------------------------


def _facts_in_docs(reply: str, docs: str) -> bool:
    if not _amounts(reply) <= _amounts(docs):
        return False
    literals = _LITERAL.findall(reply)
    if not all(_squash(lit) in _squash(docs) for lit in literals):
        return False
    rest = _LITERAL.sub(" ", _AMOUNT.sub(" ", reply))
    return set(_NUMBER.findall(rest)) <= set(_NUMBER.findall(docs))


# --- 2. Which retailer and product each fee is for -----------------------------------------


def _fees_tied_to_what_the_docs_tie_them_to(reply: str, hits: list[Hit], question: str) -> bool:
    """Each amount is said for a retailer and product the docs give it for, so one retailer's or
    product's fee is never passed off as another's or as the general fee."""
    documented = [
        mention
        for h in hits
        for mention in _fee_mentions(body(h), retailers(h.heading), products(h.heading))
    ]
    for amount, shop, item in _fee_mentions(reply, retailers(question), products(question)):
        if not any(
            amount == said
            and (not for_shop or for_shop & shop)
            and (not for_item or for_item & item)
            for said, for_shop, for_item in documented
        ):
            return False
    return True


def _fee_mentions(text: str, shop: set[str], item: set[str]) -> Iterator[FeeMention]:
    """(amount, retailers, products) for every amount in the text, judged clause by clause.

    A clause naming no retailer takes the last one named in its sentence, else any its sentence
    names later ("$20 CAD for a Booster Box on Pokémon Center Canada"), else the last one before
    it (starting from `shop`). One naming no product takes the last one named in its sentence
    (starting from `item`). Clauses rather than headings, because a fee overview names a
    retailer and product per clause.
    """
    for sentence in _sentences(text):
        named_item, named_here = item, False
        for clause in _CLAUSE.split(sentence):
            if retailers(clause):
                shop, named_here = retailers(clause), True
            elif not named_here:
                shop = retailers(sentence) or shop
            named_item = products(clause) or named_item
            for amount in _amounts(clause):
                yield amount, shop, named_item


# --- 3. Yes or no ---------------------------------------------------------------------------


def _agrees_with_quote(reply: str, quote: str) -> bool:
    """The reply contains the quoted answer and says yes or no the same way: it opens with the
    same yes or no, and its clauses are negated like the quote clauses they match.

    Compared clause by clause because one sentence can mix a no and a yes ("Amazon Prime is not
    required, but Prime accounts check out faster").
    """
    quoted, replied = _yes_or_no(quote), _yes_or_no(reply)
    if quoted is not None and replied is not None and quoted != replied:
        return False
    said = _clauses(reply)
    if not said or not terms(reply) & terms(quote):
        return False  # the reply does not contain the documented answer at all
    for clause in _clauses(quote):
        overlap, closest = max((len(terms(c) & terms(clause)), c) for c in said)
        negated_alike = bool(_NEGATION.search(closest)) == bool(_NEGATION.search(clause))
        if overlap >= _MIN_CLAUSE_OVERLAP and not negated_alike:
            return False
    return True


def _yes_or_no(text: str) -> bool | None:
    """True or False if the text opens with a yes or a no, else None."""
    opened = _OPENER.match(text)
    return None if opened is None else bool(opened.group(1))


# --- 4. No claim of its own -----------------------------------------------------------------


def _says_nothing_new(reply: str, hits: list[Hit], question: str) -> bool:
    """The reply adds no claim: no retailer the question does not ask about, few words beyond the
    sections, and every sentence mostly backed by a single doc sentence rather than words
    gathered from several. Headings count as doc sentences (restating the question is fine);
    document titles ("Bandai: ACO Guide") do not, as their words fit almost any reply."""
    asked = retailers(question)
    if asked and not retailers(reply) <= asked:
        return False
    docs = [s for h in hits for s in [h.heading, *_sentences(body(h))]] + _sentences(question)
    known = set().union(*(terms(s) for s in docs))
    if len(terms(reply) - known - _CONVERSATIONAL) > MAX_NEW_WORDS:
        return False
    for sentence in _sentences(reply):
        words = terms(sentence) - _CONVERSATIONAL
        if len(words) < 3:
            continue
        support = max(len(words & terms(d)) for d in docs) / len(words)
        if support < MIN_SENTENCE_SUPPORT:
            return False
    return True


# --- Text helpers ---------------------------------------------------------------------------


def _amounts(text: str) -> set[str]:
    return {(a or b).replace(",", "") for a, b in _AMOUNT.findall(text)}


def _sentences(text: str) -> list[str]:
    return [s for s in re.split(r"(?<=[.!?])\s+|\n+", text) if s.strip()]


def _clauses(text: str) -> list[str]:
    return [c for s in _sentences(text) for c in _CLAUSE.split(s) if terms(c)]


def _squash(text: str) -> str:
    return re.sub(r"\s+", "", text).lower().rstrip(".,)")
