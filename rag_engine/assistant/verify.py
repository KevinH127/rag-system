"""Only documented facts are ever given as an answer.

A 3B model invents plausible answers ("Yes, your card is encrypted") instead of saying it does
not know, and neither embedding distance nor word overlap can tell "answered by the docs" from
"merely related to the docs" (measured on evals/golden.jsonl). So a judge must quote the sentence
that answers the question, the quote is checked here against the section text, and only a
question with a verified section is answered at all.

The answer itself may be the model's own wording, but only if `grounded` finds nothing in it the
docs do not say: every amount, number, link and channel is in the sections, each fee is tied to
the retailer and product the docs give it for, and it agrees with the quote on yes or no.
Otherwise the verified section is sent verbatim.
"""

import re
from collections.abc import Callable, Iterator

from rag_engine.assistant.vocabulary import products, retailers, terms
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


def verbatim(hit: Hit) -> str:
    """The section as sent to a customer: its text without a bare opening "Yes." or "No.".

    That word answers the section's heading, which the customer never sees, and may have asked
    the opposite: "Can I get a refund if it arrives damaged?" got "Yes. Once an item is delivered,
    the ACO fee is owed".
    """
    return re.sub(r"^(?:Yes|No)\.\s+", "", body(hit))


def _normalise(text: str) -> str:
    return re.sub(r"[^a-z0-9$]+", " ", text.lower()).strip()


def is_supported(quote: str, text: str) -> bool:
    """True if the quote really appears in the text (ignoring case, punctuation and spacing)."""
    q = _normalise(quote)
    return len(q) >= MIN_QUOTE_CHARS and q[:_MATCH_CHARS] in _normalise(text)


def about_another_retailer(hit: Hit, question: str) -> bool:
    """True if the question names retailers and the section covers only others: its heading
    names only others, or, with no retailer in its heading, its text does.

    The judge accepted "The ACO fee for other Pokémon Center Canada items is set after the drop"
    for "How much is the ACO fee for Bandai drops?", and the general fee overview, which names
    every retailer but Bandai, for "how much is the aco fee bandai".
    """
    asked = retailers(question)
    named = retailers(hit.heading) or retailers(body(hit))
    return bool(asked and named and not asked & named)


def check_sections(
    question: str, hits: list[Hit], judge: Judge, *, general: bool = False
) -> list[Check]:
    """Judge the closest few sections in order, stopping at the first verified answer.

    Sections about a retailer the question does not ask about are never judged, and for a
    `general` answer (every retailer) sections whose heading names no retailer are judged first.
    The quote is checked against the body only, so quoting the heading back does not count.
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


def answer_of(checks: list[Check]) -> Hit | None:
    """The section that passed, if any."""
    check = verified(checks)
    return check.hit if check else None


_AMOUNT = re.compile(
    r"\$\s?(\d[\d,]*(?:\.\d+)?)|(\d[\d,]*(?:\.\d+)?)\s?(?:CAD|dollars)\b", re.IGNORECASE
)
# Links, emails, domains and #channels must appear in the docs exactly.
_LITERAL = re.compile(r"https?://\S+|[\w.+-]+@[\w-]+\.[\w.]+|#[\w-]+|\b[\w-]+\.(?:ca|com)\b\S*")
_NUMBER = re.compile(r"\d+(?:\.\d+)?")
_NEGATION = re.compile(
    r"\b(?:no|not|never|cannot|nothing|none|nope)\b|n't\b|\bcant\b", re.IGNORECASE
)


def _amounts(text: str) -> set[str]:
    return {(a or b).replace(",", "") for a, b in _AMOUNT.findall(text)}


def _sentences(text: str) -> list[str]:
    return [s for s in re.split(r"(?<=[.!?])\s+|\n+", text) if s.strip()]


def _squash(text: str) -> str:
    return re.sub(r"\s+", "", text).lower().rstrip(".,)")


_CLAUSE = re.compile(r"[,;(]|\b(?:and|while|but)\b")

FeeMention = tuple[str, set[str], set[str]]


def _fee_mentions(text: str, shop: set[str], item: set[str]) -> Iterator[FeeMention]:
    """(amount, retailers, products) for every amount in the text, judged clause by clause.

    A clause naming no retailer takes the last one named in its sentence, else any its sentence
    names later ("$20 CAD for a Booster Box (pre-order) on Pokémon Center Canada"), else the last
    one before it (starting from `shop`). One naming no product takes the last one named in its
    sentence (starting from `item`). A fee overview names a retailer and product per clause ("$25
    CAD for an Elite Trainer Box, $20 CAD for a Booster Box"), so headings alone would tie every
    amount in it to nothing.
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


def _facts_in_docs(reply: str, docs: str) -> bool:
    if not _amounts(reply) <= _amounts(docs):
        return False
    literals = _LITERAL.findall(reply)
    if not all(_squash(lit) in _squash(docs) for lit in literals):
        return False
    rest = _LITERAL.sub(" ", _AMOUNT.sub(" ", reply))
    return set(_NUMBER.findall(rest)) <= set(_NUMBER.findall(docs))


def _fees_tied_to_what_the_docs_tie_them_to(reply: str, hits: list[Hit], question: str) -> bool:
    """Each amount is said for a retailer and product the docs give it for. Measured: "The fee is
    $2.50 to $5 per item" (Walmart's fee as the general one) and "$25 CAD for pre-orders on
    Pokémon Center Canada" ($25 is the Elite Trainer Box fee; a Booster Box is $20)."""
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


_OPENER = re.compile(
    r"^\W*(?:(yes|yeah|yep|sure|absolutely)|no|nope|not|unfortunately)\b", re.IGNORECASE
)


def _yes_or_no(text: str) -> bool | None:
    """True or False if the text opens with a yes or a no, else None."""
    opened = _OPENER.match(text)
    return None if opened is None else bool(opened.group(1))


def _clauses(text: str) -> list[str]:
    return [c for s in _sentences(text) for c in _CLAUSE.split(s) if terms(c)]


def _agrees_with_quote(reply: str, quote: str) -> bool:
    """The reply contains the quoted answer and says yes or no the same way: it opens with the
    same yes or no, and a reply clause sharing at least 3 words with a clause of the quote is
    negated the same way. Measured: "Yes, you still pay the ACO fee if your order got cancelled",
    followed by the documented sentences, passed a check of the closest sentence only; and
    clauses, not sentences, because "Amazon Prime is not required, but Prime accounts check out
    faster" mixes a no and a yes. Fewer shared words pair unrelated clauses ("there is no ACO
    fee" with "the ACO fee is only charged on orders that are delivered")."""
    quoted, replied = _yes_or_no(quote), _yes_or_no(reply)
    if quoted is not None and replied is not None and quoted != replied:
        return False
    said = _clauses(reply)
    if not said or not terms(reply) & terms(quote):
        return False  # the reply does not contain the documented answer at all
    for clause in _clauses(quote):
        overlap, closest = max((len(terms(c) & terms(clause)), c) for c in said)
        if overlap >= 3 and bool(_NEGATION.search(closest)) != bool(_NEGATION.search(clause)):
            return False
    return True


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
# Measured on the golden set (60 answers in the model's own words): answers that invented a
# claim added 3-11 words found in no section ("check the 'My Orders' section of your account");
# faithful rewordings added at most 2.
MAX_NEW_WORDS = 2
# Each reply sentence must share at least this much of its wording with one doc sentence (a
# heading counts: restating the question is fine). Invented sentences measured 0.17-0.25 ("you
# can also contact Walmart's support for further assistance"); faithful ones 0.38 and up.
MIN_SENTENCE_SUPPORT = 0.34


def _says_nothing_new(reply: str, hits: list[Hit], question: str) -> bool:
    """The reply adds no claim: no retailer the question does not ask about, few words beyond the
    sections, and every sentence mostly backed by a single doc sentence rather than words
    gathered from several. Headings count, document titles do not: "Bandai: ACO Guide" backed an
    invented "find the correct region on the P-Bandai website or in the ACO Guide"."""
    asked = retailers(question)
    if asked and not retailers(reply) <= asked:
        return False  # measured: a Costco jig question got Walmart's jig guidance
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


def grounded(reply: str, answer: Check, hits: list[Hit], question: str) -> bool:
    """True if the model's own answer says nothing the docs do not (see the module docstring)."""
    if not reply.strip() or not answer.quote:
        return False
    docs = "\n".join(body(h) for h in hits)
    return (
        _facts_in_docs(reply, docs)
        and _fees_tied_to_what_the_docs_tie_them_to(reply, hits, question)
        and _agrees_with_quote(reply, answer.quote)
        and _says_nothing_new(reply, hits, question)
    )


def find_answer(question: str, hits: list[Hit], judge: Judge) -> Hit | None:
    """The closest section the judge can quote an answer from, checking the top few in order."""
    return answer_of(check_sections(question, hits, judge))
