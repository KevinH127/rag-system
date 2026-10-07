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

CLARIFY_FALLBACK = (
    "Could you tell me a bit more about what you need help with, "
    "such as which retailer it is about?"
)

SECRET_WARNING = (
    "For your safety, please don't share passwords, one-time codes or card numbers here. "
    "I removed it from your message. If staff need a code, they will ask in your account login "
    "ticket."
)

OFF_TOPIC = (
    "I can only help with questions about Trevona ACO, such as fees, profiles, retailers and "
    "support tickets. Please only ask me about Trevona."
)

ASK_AGAIN = "What can I help you with?"

# Sent for anything after a handoff: staff own the conversation from then on.
CLOSED = (
    "This conversation has been passed to a team member, so I can't continue it here. Please "
    "reply in your support ticket (or open one in #make-a-ticket)."
)

WHICH_RETAILER = (
    "Which retailer is this about: Pokémon Center Canada, Walmart Canada, Amazon Canada or "
    "Costco Canada?"
)
