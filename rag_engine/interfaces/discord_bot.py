"""The Discord bot: answers customers inside support-ticket channels.

It answers in every channel it can read, so give it access to ticket channels only. Each channel
is one conversation (one Engine, logged under the channel name "discord"). After a handoff, or as
soon as a member of the staff role posts in a ticket, the bot stays silent there: staff own the
ticket from then on.

`Tickets` and `messages_for` hold the logic and need no Discord connection, so they are tested
directly; `run` only wires them to Discord.
"""

import asyncio
import logging
from collections import defaultdict
from collections.abc import Callable

import discord

from rag_engine.assistant import replies
from rag_engine.assistant.engine import Engine
from rag_engine.assistant.redact import redact
from rag_engine.config import settings
from rag_engine.logbook import store
from rag_engine.models import Action, Intent, Response

log = logging.getLogger(__name__)

# Discord rejects longer messages.
MAX_MESSAGE_CHARS = 2000


class Tickets:
    """One conversation per ticket channel, and the tickets the bot no longer answers in."""

    def __init__(self, new_engine: Callable[[], Engine]) -> None:
        self._new_engine = new_engine
        self._engines: dict[int, Engine] = {}
        self._handed_over: set[int] = set()

    def respond(self, channel_id: int, text: str) -> Response | None:
        """The bot's response to a customer message, or None once staff own the ticket."""
        if channel_id in self._handed_over:
            return None
        engine = self._engines.get(channel_id)
        if engine is None:
            engine = self._engines[channel_id] = self._new_engine()
        response = engine.respond(text)
        if engine.closed:
            self.hand_over(channel_id)
        return response

    def hand_over(self, channel_id: int) -> None:
        """Staff own the ticket from now on: the bot stays silent in it."""
        self._handed_over.add(channel_id)
        self._engines.pop(channel_id, None)

    def forget(self, channel_id: int) -> None:
        """The ticket channel was deleted."""
        self._handed_over.discard(channel_id)
        self._engines.pop(channel_id, None)


def messages_for(response: Response, staff_role_id: int | None) -> list[str]:
    """The Discord messages to post: the reply, then on a handoff the ticket summary for staff,
    pinging their role. Each fits Discord's length limit."""
    messages = split(response.reply)
    if response.action is Action.HANDOFF and response.summary:
        ping = f"<@&{staff_role_id}> " if staff_role_id else ""
        messages += split(f"{ping}**Ticket summary:** {response.summary}")
    return messages


def split(text: str, limit: int = MAX_MESSAGE_CHARS) -> list[str]:
    """The text in pieces of at most `limit` characters, broken at a line end or space."""
    pieces: list[str] = []
    while len(text) > limit:
        cut = max(text.rfind("\n", 0, limit), text.rfind(" ", 0, limit))
        if cut <= 0:
            cut = limit
        pieces.append(text[:cut].rstrip())
        text = text[cut:].lstrip()
    if text:
        pieces.append(text)
    return pieces


def run() -> None:
    """Connect to Discord and answer in ticket channels until stopped (Ctrl+C)."""
    if settings.discord_token is None:
        raise SystemExit("Set DISCORD_TOKEN in .env first (see the README, 'Discord bot').")
    tickets = Tickets(lambda: Engine(recorder=store.recorder("discord")))
    client = create_client(tickets, settings.discord_staff_role_id)
    client.run(settings.discord_token.get_secret_value())


def create_client(tickets: Tickets, staff_role_id: int | None) -> discord.Client:
    """A Discord client that answers customers through `tickets`."""
    intents = discord.Intents.default()
    intents.message_content = True  # also switch on "Message Content Intent" in the portal
    client = discord.Client(intents=intents)
    # Messages in one ticket are answered one at a time, in order.
    locks: defaultdict[int, asyncio.Lock] = defaultdict(asyncio.Lock)
    mentions = discord.AllowedMentions(everyone=False, users=False, roles=True)

    @client.event
    async def on_ready() -> None:
        log.info("Logged in to Discord as %s", client.user)

    @client.event
    async def on_message(message: discord.Message) -> None:
        if message.author.bot or message.guild is None or not message.content.strip():
            return  # other bots (the ticket bot included), DMs, attachment-only messages
        channel = message.channel
        async with locks[channel.id]:
            if _is_staff(message.author, staff_role_id):
                tickets.hand_over(channel.id)
                return
            try:
                async with channel.typing():
                    # The engine blocks on Ollama and Postgres; a thread keeps Discord connected.
                    response = await asyncio.to_thread(tickets.respond, channel.id, message.content)
            except Exception:
                log.exception("Could not answer in channel %s", channel.id)
                tickets.hand_over(channel.id)
                response = _unavailable(message.content)
            if response is None:
                return
            for text in messages_for(response, staff_role_id):
                await channel.send(text, allowed_mentions=mentions)

    @client.event
    async def on_guild_channel_delete(channel: discord.abc.GuildChannel) -> None:
        tickets.forget(channel.id)
        locks.pop(channel.id, None)

    return client


def _is_staff(author: discord.abc.User, staff_role_id: int | None) -> bool:
    roles = getattr(author, "roles", ())
    return staff_role_id is not None and any(r.id == staff_role_id for r in roles)


def _unavailable(message: str) -> Response:
    """A handoff for when the engine fails (Ollama or Postgres down)."""
    return Response(
        Intent.QUESTION, Action.HANDOFF, replies.UNAVAILABLE, summary=redact(message).text
    )
