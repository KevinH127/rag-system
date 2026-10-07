"""Customer-facing texts the engine writes itself, so the LLM cannot word them wrongly.

Handoff and off-topic replies live here because a small model, left to write them, promised
actions the bot cannot perform ("I'll create a ticket and email you").
"""

NO_ANSWER = (
    "I'm not able to answer that myself, so I'll pass it to a team member. "
    "Please open or reply in a support ticket so they can help."
)

REQUEST_HANDOFF = (
    "I can't do that myself, so a team member needs to help. Please open a support ticket in "
    "#make-a-ticket (or reply in the ticket you already have open) and include the details below."
)

SECRET_WARNING = (
    "For your safety, please don't share passwords, one-time codes or card numbers here. "
    "I removed it from your message. If staff need a code, they will ask in your account login "
    "ticket."
)

# Sent when a message was nothing but a secret, which was removed.
ASK_AGAIN = "What can I help you with?"

_HELP_WITH = (
    "I can help with anything about Trevona ACO: fees, profiles, retailers, drops and support "
    "tickets."
)

OFF_TOPIC = (
    "Sorry, I can only help with questions about Trevona ACO, such as fees, profiles, retailers, "
    "drops and support tickets. Is there anything about Trevona I can help with?"
)

# Pleasantries (vocabulary.small_talk) get a friendly reply that steers back to Trevona.
SMALL_TALK = {
    "greeting": f"Hi! {_HELP_WITH} {ASK_AGAIN}",
    "how_are_you": f"I'm doing well, thanks for asking! {_HELP_WITH} {ASK_AGAIN}",
    "thanks": "You're welcome! Let me know if there's anything else I can help with.",
    "bye": "Thanks for reaching out, and have a great day!",
}

# Sent when the bot cannot reach its model or database (an interface catches the error).
UNAVAILABLE = (
    "Sorry, I'm having trouble answering right now, so I'll pass this to a team member. They "
    "will reply here."
)

# Sent for anything after a handoff: staff own the conversation from then on.
CLOSED = (
    "This conversation has been passed to a team member, so I can't continue it here. Please "
    "reply in your support ticket (or open one in #make-a-ticket)."
)
