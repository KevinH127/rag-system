"""Word-level normalisation for deciding whether a message is about Trevona ACO at all.

Accents removed, stop words dropped, light stemming ("profiles" and "profile" match).
"""

import re
import unicodedata

# Kept alongside the original word so either form can match.
_ALIASES = {"dm": "direct message", "dms": "dm direct message", "etransfer": "transfer"}


def _words(text: str) -> frozenset[str]:
    return frozenset(text.split())


_STOP_WORDS = _words(
    """
    a an the i me my we you your our is are was am do does did can could would should will to of
    in on at for and or but if so it its this that these those what why how when where who which
    get got have has had be been please just now with about as by from not no there much
    """
)


def _stem(word: str) -> str:
    for suffix in ("ing", "ed", "s"):
        if suffix == "s" and word.endswith("ss"):  # address, pass
            break
        if word.endswith(suffix) and len(word) - len(suffix) >= 3:
            word = word[: -len(suffix)]
            break
    # "receive"/"received" and "boxe(s)"/"box" end up identical.
    return word[:-1] if word.endswith("e") and len(word) > 3 else word


def terms(text: str) -> set[str]:
    """Normalised content words of a text."""
    plain = unicodedata.normalize("NFKD", text).encode("ascii", "ignore").decode().lower()
    words: list[str] = []
    for word in re.findall(r"[a-z0-9]+", plain):
        words.append(word)
        words.extend(_ALIASES.get(word, "").split())
    return {_stem(w) for w in words if len(w) > 1 and w not in _STOP_WORDS}


# Any of these in a message means it is plausibly about Trevona ACO.
_DOMAIN = frozenset(
    terms(
        """
        trevona aco fee profile ticket drop checkout retailer amazon walmart costco pokemon bandai
        dashboard discord shipment shipping address jig cutoff membership imap otp order account
        vault venn booster etb elite trainer bundle queue role payment card announcement team
        staff dm money pay paid transfer bot server channel delivery tracking refund cancel
        work working problem issue broken error wrong stuck fail help
        """
    )
)


def mentions_trevona(text: str) -> bool:
    """True if the text uses any word tied to Trevona ACO's domain."""
    return bool(terms(text) & _DOMAIN)


# Words naming what a question is about rather than what it asks. The docs repeat a question per
# retailer or product ("How much does Trevona charge (ACO fee) for Walmart Canada items?").
_RETAILERS = frozenset(terms("walmart amazon costco pokemon bandai"))
_SUBJECTS = _RETAILERS | frozenset(terms("etb elite trainer booster box bundle"))
# Words that can come with a name without asking anything ("Pokémon Centre", "Costco then?").
_FILLERS = frozenset(terms("center centre canada then too also instead one ok okay"))


def retailers(text: str) -> set[str]:
    """The retailers a text names."""
    return terms(text) & _RETAILERS


# Products the docs give their own fee. Matched as phrases: "box" alone could be either box.
_PRODUCTS = (
    ("etb", re.compile(r"\betbs?\b|\belite trainer box")),
    ("booster box", re.compile(r"\bbooster box")),
    ("booster bundle", re.compile(r"\bbundle")),
)


def products(text: str) -> set[str]:
    """The products a text names, as "etb", "booster box" or "booster bundle"."""
    plain = unicodedata.normalize("NFKD", text).encode("ascii", "ignore").decode().lower()
    return {name for name, pattern in _PRODUCTS if pattern.search(plain)}


def names_subject(text: str) -> bool:
    """True if the text names a retailer or product."""
    return bool(terms(text) & _SUBJECTS)


def only_names_subject(text: str) -> bool:
    """True for a message that names a retailer or product and asks nothing ("what about Costco?")."""
    words = terms(text)
    return bool(words & _SUBJECTS) and words <= _SUBJECTS | _FILLERS


# Words that widen a question to every retailer instead of naming one ("in general").
_GENERAL = frozenset(
    terms("general generally overall overview everything every all both particular")
)


def asks_in_general(text: str) -> bool:
    """True if the text asks about every retailer rather than one ("for all of them")."""
    return bool(terms(text) & _GENERAL)


def is_follow_up(text: str) -> bool:
    """True for a message that only narrows or widens the previous question and asks nothing new
    ("what about Costco?", "in general", "for all of them")."""
    words = terms(text)
    return bool(words) and words <= _SUBJECTS | _FILLERS | _GENERAL | frozenset(terms("them any"))


def without_subjects(text: str) -> str:
    """The text with retailer and product names removed, keeping what it asks."""
    return " ".join(w for w in text.split() if not (terms(w) and terms(w) <= _SUBJECTS | _FILLERS))


# A message made only of these words asks nothing; it is a pleasantry.
_PLEASANTRIES = frozenset(
    terms(
        """
        hi hello hey yo sup good morning afternoon evening doing going today things thanks thank
        thx ty cheers appreciate bye goodbye see ya cya have day night one great fine well ok okay
        awesome nice cool lot
        """
    )
)
# First match wins: "hi, how are you?" is asked how it is doing, "hey thanks" is thanked.
_SMALL_TALK = (
    ("how_are_you", re.compile(r"\bhow (?:are|r) (?:you|u)\b|\bhow'?s it going\b")),
    ("thanks", re.compile(r"\b(?:thanks?|thank you|thx|ty|cheers|appreciate)\b")),
    ("bye", re.compile(r"\b(?:bye|goodbye|see ya|cya|have a (?:good|great) (?:one|day|night))\b")),
    ("greeting", re.compile(r"\b(?:hi|hello|hey|yo|sup|good (?:morning|afternoon|evening))\b")),
)


def small_talk(text: str) -> str | None:
    """The kind of pleasantry a message is ("greeting", "how_are_you", "thanks", "bye"), or None
    if it asks anything."""
    if not terms(text) <= _PLEASANTRIES:
        return None
    plain = unicodedata.normalize("NFKD", text).encode("ascii", "ignore").decode().lower()
    return next((kind for kind, pattern in _SMALL_TALK if pattern.search(plain)), None)
