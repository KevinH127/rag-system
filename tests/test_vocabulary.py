import pytest

from rag_engine.assistant.vocabulary import (
    asks_in_general,
    is_follow_up,
    mentions_trevona,
    names_subject,
    only_names_subject,
    small_talk,
    terms,
    without_subjects,
)


@pytest.mark.parametrize(
    "msg",
    [
        "What's the Bandai fee?",
        "my PROFILE is inactive",
        "Pokémon drop",
        # Words that were meant to be domain terms but were silently missing before.
        "Can I just DM one of the team members?",
        "Can I just send you the money by e-transfer?",
    ],
)
def test_domain_terms_detected(msg):
    assert mentions_trevona(msg)


@pytest.mark.parametrize(
    "msg", ["Write me a python script", "what's the weather", "tell me a joke"]
)
def test_unrelated_text_has_no_terms(msg):
    assert not mentions_trevona(msg)


def test_plurals_and_tenses_normalise_to_the_same_term():
    assert terms("profiles received boxes") == terms("profile receive box")


def test_accents_and_stop_words_are_ignored():
    assert terms("How do I jig my Pokémon address?") == {"jig", "pokemon", "address"}


def test_dm_alias_keeps_the_original_word():
    assert {"dm", "direct"} <= terms("can I DM you")


@pytest.mark.parametrize(
    "msg",
    [
        "what about for costco",
        "and Amazon?",
        "Costco?",
        "what about Pokémon Centre then",
        "ok what about booster boxes",
    ],
)
def test_message_that_only_names_a_retailer_or_product(msg):
    assert only_names_subject(msg)


@pytest.mark.parametrize(
    "msg",
    [
        "how do I sign up for costco?",
        "How many orders can I place with one Costco membership?",
        "what about that?",
        "thanks",
        "how much is the aco fee",
    ],
)
def test_message_that_asks_something(msg):
    assert not only_names_subject(msg)


def test_subject_names_are_detected_and_removed():
    assert names_subject("How much is the ACO fee for an Elite Trainer Box?")
    assert not names_subject("how much is the aco fee")
    assert without_subjects("how much is the aco fee walmart") == "how much is the aco fee"
    assert without_subjects("fee for Pokémon Center booster boxes?") == "fee for"


@pytest.mark.parametrize(
    "msg", ["in general", "for all of them", "what about Costco?", "not one in particular", "both"]
)
def test_message_that_only_narrows_or_widens_the_last_question_is_a_follow_up(msg):
    assert is_follow_up(msg)


@pytest.mark.parametrize("msg", ["how do I sign up in general?", "thanks", "the fee", ""])
def test_message_that_asks_something_new_is_not_a_follow_up(msg):
    assert not is_follow_up(msg)


def test_asking_in_general():
    assert asks_in_general("for all of them") and asks_in_general("in general")
    assert not asks_in_general("for costco") and not asks_in_general("how much is the fee")


@pytest.mark.parametrize(
    ("msg", "kind"),
    [
        ("hi", "greeting"),
        ("Hey there!", "greeting"),
        ("good morning", "greeting"),
        ("how are you doing today?", "how_are_you"),
        ("hi, how's it going", "how_are_you"),
        ("thanks!", "thanks"),
        ("thank you so much", "thanks"),
        ("ok cool, appreciate it", "thanks"),
        ("bye", "bye"),
        ("have a great day", "bye"),
    ],
)
def test_pleasantries_are_small_talk(msg, kind):
    assert small_talk(msg) == kind


@pytest.mark.parametrize(
    "msg", ["hi, how much is the aco fee?", "thanks, what about costco?", "ok", "in general", ""]
)
def test_a_message_that_asks_anything_is_not_small_talk(msg):
    assert small_talk(msg) is None
