import asyncio
from contextlib import asynccontextmanager
from datetime import UTC, datetime, timedelta
from types import SimpleNamespace

from helpers import section

from rag_engine.assistant import replies
from rag_engine.assistant.engine import Engine
from rag_engine.interfaces.discord_bot import (
    MAX_MESSAGE_CHARS,
    Tickets,
    create_client,
    messages_for,
    split,
)
from rag_engine.models import Action, Decision, Intent, Response

FEE = section("How much is the fee?", "The ACO fee is $25 CAD, paid after delivery.", 0.2)


def engine_that(action):
    """An Engine whose model always proposes `action`, answerable from the docs only if that
    action is an answer."""
    return Engine(
        retriever=lambda q: [FEE],
        decider=lambda s, h: Decision(intent=Intent.QUESTION, action=action, reply="x"),
        judge=lambda q, p: p.partition("\n\n")[2] if action is Action.ANSWER else None,
        summarizer=lambda messages: "Asks something.",
    )


def test_each_ticket_is_its_own_conversation():
    made = []
    tickets = Tickets(lambda: made.append(engine_that(Action.ANSWER)) or made[-1])
    tickets.respond(1, "fee?")
    tickets.respond(1, "and the fee?")
    tickets.respond(2, "fee?")
    assert len(made) == 2 and len(made[0].history) == 4


def test_after_a_handoff_the_bot_is_silent_in_that_ticket():
    tickets = Tickets(lambda: engine_that(Action.HANDOFF))
    assert tickets.respond(1, "refund me").action is Action.HANDOFF
    assert tickets.respond(1, "hello?") is None


def test_after_staff_take_over_the_bot_is_silent_until_the_ticket_is_deleted():
    tickets = Tickets(lambda: engine_that(Action.ANSWER))
    tickets.hand_over(1)
    assert tickets.respond(1, "fee?") is None
    tickets.forget(1)
    assert tickets.respond(1, "fee?").action is Action.ANSWER


def test_a_handoff_pings_the_staff_role_with_the_ticket_summary():
    handoff = Response(Intent.REQUEST, Action.HANDOFF, replies.REQUEST_HANDOFF, summary="Refund.")
    assert messages_for(handoff, 42) == [
        replies.REQUEST_HANDOFF,
        "<@&42> **Ticket summary:** Refund.",
    ]
    assert messages_for(handoff, None)[1] == "**Ticket summary:** Refund."


def test_an_answer_is_posted_alone():
    answer = Response(Intent.QUESTION, Action.ANSWER, "It's $25 CAD.")
    assert messages_for(answer, 42) == ["It's $25 CAD."]


def test_long_replies_are_split_at_word_breaks_to_fit_discord():
    text = "word " * 1000
    pieces = split(text)
    assert all(len(p) <= MAX_MESSAGE_CHARS for p in pieces)
    assert " ".join(pieces).split() == text.split()
    assert split("short") == ["short"]


# --- The Discord client, driven with fake messages -----------------------------------------


STARTED = datetime(2026, 10, 8, 12, 0, tzinfo=UTC)


class FakeChannel:
    def __init__(self, channel_id=1, created_at=STARTED + timedelta(minutes=5)):
        self.id = channel_id
        self.created_at = created_at
        self.sent = []

    @asynccontextmanager
    async def typing(self):
        yield

    async def send(self, text, allowed_mentions=None):
        self.sent.append(text)


def message(text, channel, *, bot=False, role_ids=()):
    author = SimpleNamespace(bot=bot, roles=[SimpleNamespace(id=r) for r in role_ids])
    return SimpleNamespace(content=text, author=author, channel=channel, guild=object())


def deliver(client, *messages):
    async def all_of_them():
        for m in messages:
            await client.on_message(m)

    asyncio.run(all_of_them())


def test_a_customer_message_is_answered_in_its_ticket():
    channel = FakeChannel()
    client = create_client(
        Tickets(lambda: engine_that(Action.ANSWER)), staff_role_id=42, started_at=STARTED
    )
    deliver(client, message("fee?", channel))
    assert channel.sent == ["The ACO fee is $25 CAD, paid after delivery."]


def test_bots_are_ignored_and_staff_messages_silence_the_ticket():
    channel = FakeChannel()
    client = create_client(
        Tickets(lambda: engine_that(Action.ANSWER)), staff_role_id=42, started_at=STARTED
    )
    deliver(
        client,
        message("Welcome to your ticket!", channel, bot=True),
        message("Hi, a team member here.", channel, role_ids=[42]),
        message("fee?", channel),
    )
    assert channel.sent == []


def test_a_handoff_posts_the_reply_then_pings_staff():
    channel = FakeChannel()
    client = create_client(
        Tickets(lambda: engine_that(Action.HANDOFF)), staff_role_id=42, started_at=STARTED
    )
    deliver(client, message("can I use a prepaid card?", channel), message("hello?", channel))
    assert channel.sent == [replies.NO_ANSWER, "<@&42> **Ticket summary:** Asks something."]


def test_when_the_engine_fails_staff_take_over_without_seeing_secrets():
    def broken(*_):
        raise ConnectionError("Ollama is down")

    engine = Engine(retriever=broken, decider=broken, judge=broken, summarizer=broken)
    channel = FakeChannel()
    client = create_client(Tickets(lambda: engine), staff_role_id=42, started_at=STARTED)
    deliver(client, message("my password is hunter22, why did checkout fail?", channel))
    assert channel.sent[0] == replies.UNAVAILABLE
    assert channel.sent[1].startswith("<@&42>") and "hunter22" not in channel.sent[1]


def test_tickets_opened_before_the_bot_started_are_left_to_staff():
    old_ticket = FakeChannel(created_at=STARTED - timedelta(hours=1))
    client = create_client(
        Tickets(lambda: engine_that(Action.ANSWER)), staff_role_id=42, started_at=STARTED
    )
    deliver(client, message("fee?", old_ticket))
    assert old_ticket.sent == []
