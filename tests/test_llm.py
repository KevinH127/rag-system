"""The model's JSON can be cut off or malformed; that must never crash a conversation."""

from types import SimpleNamespace

import pytest

from rag_engine.assistant import llm
from rag_engine.models import Action


def reply_with(monkeypatch, content: str) -> None:
    fake = SimpleNamespace(
        chat=lambda **_: SimpleNamespace(message=SimpleNamespace(content=content))
    )
    monkeypatch.setattr(llm, "_client", fake)


@pytest.mark.parametrize("content", ['{"intent": "question", "action": "ans', "not json", "{}"])
def test_unreadable_decision_becomes_a_handoff(monkeypatch, content):
    reply_with(monkeypatch, content)
    assert llm.decide("system", []).action is Action.HANDOFF


def test_cut_off_quote_counts_as_no_answer(monkeypatch):
    reply_with(monkeypatch, '{"quote": "The team announced three ticket types on 23 Sep')
    assert llm.quote_answer("q", "passage") is None


def test_summary_is_trimmed_and_an_empty_one_is_none(monkeypatch):
    reply_with(monkeypatch, '  "Customer asks whether prepaid cards work."  ')
    assert (
        llm.summarize(["can I use a prepaid card?"]) == "Customer asks whether prepaid cards work."
    )
    reply_with(monkeypatch, "  ")
    assert llm.summarize(["?"]) is None


def test_quote_is_returned_only_for_an_answered_verdict(monkeypatch):
    reply_with(
        monkeypatch,
        '{"quote": "No. Vault cards are not accepted.", "verdict": "passage_answers_question"}',
    )
    assert llm.quote_answer("q", "p") == "No. Vault cards are not accepted."
    reply_with(monkeypatch, '{"quote": "Something.", "verdict": "answer_not_in_passage"}')
    assert llm.quote_answer("q", "p") is None
