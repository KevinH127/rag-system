from helpers import section

from rag_engine.assistant.grounding import grounded
from rag_engine.assistant.verify import body
from rag_engine.models import Check

FEE = section("What is the Walmart fee?", "Most Walmart items cost $2.50 to $5 per item.", 0.2)
CARDS = section("Can I use a Vault card?", "No. Vault or Venn cards are not accepted.", 0.2)
LINKS = section(
    "How do I sign up?", "Open a ticket in #make-a-ticket or go to trevona.ca/dashboard.", 0.2
)
OVERVIEW = section(
    "How much is the ACO fee in general?",
    "On Pokémon Center Canada an Elite Trainer Box is $25 CAD. Walmart Canada items are mostly "
    "$2.50 to $5. Costco Canada fees are set after the drop.",
    0.2,
)
PKC_FEES = section(
    "How much is the ACO fee in general?",
    "On Pokémon Center Canada, pre-orders are $25 CAD for an Elite Trainer Box, $20 CAD for a "
    "Booster Box and $5 CAD for a Booster Bundle. On Walmart Canada, most items are $2.50 to $5.",
    0.2,
)
BOX = section(
    "How much is the ACO fee for a Booster Box on Pokémon Center Canada?",
    "Trevona charges an ACO fee of $20 CAD for a Booster Box (pre-order) on Pokémon Center "
    "Canada, paid after delivery.",
    0.2,
)
CANCELLED = section(
    "Do I pay the ACO fee if my order is cancelled?",
    "No. The ACO fee is only charged on orders that are delivered. If the retailer cancels an "
    "order, there is no ACO fee.",
    0.2,
)
PRIME = section(
    "Do I need Amazon Prime?",
    "Amazon Prime is not required, but Prime accounts generally see higher success rates.",
    0.2,
)
SHIPMENT = section(
    "I received my shipment. What next?",
    "Reply in your support ticket or ACO fee ticket so the team can confirm what you owe.",
    0.2,
)

GENERAL = "how much is the aco fee in general"


def ok(reply, hit, question="q"):
    """grounded() with the whole section body as the judge's quote."""
    return grounded(reply, Check(hit, body(hit), True), [hit], question)


def test_faithful_rewording_holds_up():
    assert ok("For Walmart, most items cost $2.50 to $5 each.", FEE, "walmart fee?")
    assert ok("Unfortunately Vault and Venn cards aren't accepted.", CARDS)
    assert ok("Sure! Just open a ticket in #make-a-ticket.", LINKS)


def test_reply_without_the_documented_answer_is_rejected():
    assert not ok("Happy to help with that!", FEE, "walmart fee?")
    assert not ok("", FEE)


# --- 1. Amounts, numbers, links ------------------------------------------------------------


def test_amount_or_number_the_docs_lack_is_rejected():
    assert not ok("For Walmart, most items cost $2.50 to $6.", FEE, "walmart fee?")
    assert not ok("For Walmart, most items cost $3 each.", FEE, "walmart fee?")
    assert not ok("For Walmart, it takes 3 days, at $2.50 to $5.", FEE, "walmart fee?")


def test_link_or_channel_the_docs_lack_is_rejected():
    assert not ok("Open a ticket in #support.", LINKS)
    assert not ok("Go to trevona.ca/signup to sign up.", LINKS)


# --- 2. Which retailer and product each fee is for -----------------------------------------


def test_retailer_fee_passed_off_as_the_general_fee_is_rejected():
    assert not ok("The ACO fee is $2.50 to $5 per item.", FEE, "what is the aco fee")
    assert ok("The ACO fee is $2.50 to $5 per item.", FEE, "what is the walmart fee")


def test_fee_said_for_the_wrong_retailer_is_rejected():
    assert not ok("On Costco Canada most items are $2.50 to $5.", FEE, "fee?")


def test_overview_amounts_keep_the_retailer_their_sentence_names():
    faithful = (
        "It depends on the retailer. On Pokémon Center Canada an Elite Trainer Box is $25 CAD, "
        "and Walmart Canada items are mostly $2.50 to $5."
    )
    assert ok(faithful, OVERVIEW, GENERAL)
    assert not ok("The ACO fee is $25 CAD per item.", OVERVIEW, GENERAL)
    assert not ok("On Walmart Canada it's $25 CAD.", OVERVIEW, GENERAL)


def test_retailer_carries_over_to_the_next_sentence():
    reply = "On Pokémon Center Canada an Elite Trainer Box is $25 CAD. Costco fees come later."
    assert ok(reply, OVERVIEW, "fees in general")
    assert not ok("Costco fees come later. An Elite Trainer Box is $25 CAD.", OVERVIEW, "fees?")


def test_retailer_named_after_the_fee_in_the_same_sentence_counts():
    reply = "It's $20 CAD for a Booster Box (pre-order) on Pokémon Center Canada."
    assert ok(reply, BOX, "booster box pre-order fee?")


def test_product_fee_passed_off_as_every_pre_orders_fee_is_rejected():
    assert not ok("It's $25 CAD for pre-orders on Pokémon Center Canada.", PKC_FEES, GENERAL)
    faithful = "On Pokémon Center Canada it's $25 CAD for an ETB and $20 CAD for a Booster Box."
    assert ok(faithful, PKC_FEES, GENERAL)


def test_fees_swapped_within_one_sentence_are_rejected():
    swapped = "On Pokémon Center Canada it's $20 CAD for an ETB and $25 CAD for a Booster Box."
    assert not ok(swapped, PKC_FEES, GENERAL)
    shops = "Pokémon Center Canada charges $2.50 to $5, and Walmart Canada $25 CAD for an ETB."
    assert not ok(shops, PKC_FEES, GENERAL)


def test_amount_shared_by_two_fees_matches_either_but_not_a_third():
    # $5 is both the Booster Bundle fee and the top of Walmart's range.
    assert ok("A Booster Bundle on Pokémon Center Canada is $5 CAD.", PKC_FEES, GENERAL)
    assert ok("Walmart Canada items are mostly $2.50 to $5.", PKC_FEES, GENERAL)
    assert not ok("An ETB on Pokémon Center Canada is $5 CAD.", PKC_FEES, GENERAL)


# --- 3. Yes or no ---------------------------------------------------------------------------


def test_flipped_yes_or_no_is_rejected():
    assert not ok("Yes, Vault and Venn cards are accepted.", CARDS)
    assert not ok("Most Walmart items don't cost $2.50 to $5.", FEE, "walmart fee?")


def test_opening_yes_against_a_quoted_no_is_rejected_even_if_the_rest_is_copied():
    copied = (
        "Yes, you still pay the ACO fee if your order got cancelled. The ACO fee is only charged "
        "on orders that are delivered."
    )
    assert not ok(copied, CANCELLED, "do I pay if it got cancelled?")
    assert ok("No, the ACO fee is only charged on orders that are delivered.", CANCELLED)


def test_quote_mixing_a_no_and_a_yes_is_matched_clause_by_clause():
    reply = "No, you don't need Amazon Prime. However, Prime accounts generally see higher success."
    assert ok(reply, PRIME, "do I need prime?")
    assert not ok("Yes, Amazon Prime is required.", PRIME, "do I need prime?")


# --- 4. No claim of its own -----------------------------------------------------------------


def test_reply_naming_a_retailer_the_question_does_not_ask_about_is_rejected():
    walmart = section("Can I jig for Walmart Canada?", "No guidance was given for Walmart.", 0.2)
    assert not ok("No guidance was given for Walmart.", walmart, "can I jig for costco?")


def test_invented_claims_are_rejected_but_conversational_wording_is_not():
    invented = (
        "Reply in your support ticket so the team can confirm what you owe. Please check your "
        "order details in the My Orders section of your account to confirm quantities."
    )
    assert not ok(invented, SHIPMENT, "I got my shipment")
    friendly = "Great! Just reply in your support ticket so the team can confirm what you owe."
    assert ok(friendly, SHIPMENT, "I got my shipment")
