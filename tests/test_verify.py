from rag_engine.assistant.verify import (
    body,
    check_sections,
    find_answer,
    grounded,
    is_supported,
    verbatim,
)
from rag_engine.models import Check, Hit


def section(heading: str, text: str, distance: float) -> Hit:
    return Hit("doc.md", heading, f"Doc > {heading}\n\n{text}", distance)


IMAP = section("Why is IMAP required?", "IMAP must be set up correctly or checkouts fail.", 0.3)
FEE = section("What is the Walmart fee?", "Most Walmart items cost $2.50 to $5 per item.", 0.2)


def test_body_drops_the_label_line():
    assert body(FEE) == "Most Walmart items cost $2.50 to $5 per item."


def test_quote_matches_despite_case_punctuation_and_spacing():
    assert is_supported("most walmart items cost $2.50  to $5", body(FEE))


def test_invented_or_too_short_quotes_are_rejected():
    assert not is_supported("Walmart items are free", body(FEE))
    assert not is_supported("Most", body(FEE))
    assert not is_supported("", body(FEE))


def test_quoting_the_heading_does_not_count_as_an_answer():
    def judge(question, passage):
        return "What is the Walmart fee?"

    assert find_answer("walmart fee?", [FEE], judge) is None


def test_first_verified_section_in_distance_order_wins():
    def judge(question, passage):
        return passage.partition("\n\n")[2]

    assert find_answer("q", [IMAP, FEE], judge) == FEE  # FEE is closer


def test_falls_through_to_the_next_section_when_the_closest_has_no_answer():
    def judge(question, passage):
        return "IMAP must be set up correctly" if "IMAP" in passage else None

    assert find_answer("q", [IMAP, FEE], judge) == IMAP


def test_only_the_closest_few_sections_are_checked(monkeypatch):
    from rag_engine.config import settings

    monkeypatch.setattr(settings, "verify_top_n", 1)
    checked = []
    find_answer("q", [IMAP, FEE], lambda q, p: checked.append(p))
    assert checked == [FEE.content]


def test_checks_record_rejected_quotes_and_stop_at_the_first_verified_section():
    def judge(question, passage):
        if "Walmart" in passage:
            return "Walmart items are free"  # invented: not in the section
        return "IMAP must be set up correctly"

    checks = check_sections("q", [IMAP, FEE], judge)
    assert [(c.hit, c.quote, c.supported) for c in checks] == [
        (FEE, "Walmart items are free", False),
        (IMAP, "IMAP must be set up correctly", True),
    ]


def test_no_checks_after_a_verified_section():
    def judge(question, passage):
        return passage.partition("\n\n")[2]

    assert [c.hit for c in check_sections("q", [IMAP, FEE], judge)] == [FEE]


# --- Sections about another retailer ---------------------------------------------------------

PKC_OTHER = section(
    "How much is the ACO fee for other Pokémon Center Canada items?",
    "The ACO fee for other Pokémon Center Canada items is set after the drop.",
    0.25,
)


def test_section_about_another_retailer_is_never_judged():
    # Reported: the Pokémon Center fee was given for "How much is the ACO fee for Bandai drops?".
    checked = []
    check_sections("ACO fee for Bandai drops?", [PKC_OTHER, FEE], lambda q, p: checked.append(p))
    assert checked == []


def test_general_section_whose_text_covers_only_other_retailers_is_never_judged():
    # Reported: the general fee overview, which names every retailer but Bandai, answered "bandai".
    overview = section(
        "How much is the ACO fee in general?",
        "Walmart Canada is $2.50 to $5. Costco is later.",
        0.2,
    )
    checked = []
    check_sections("aco fee bandai", [overview], lambda q, p: checked.append(p))
    assert checked == []
    check_sections("aco fee costco", [overview], lambda q, p: checked.append(p))
    assert checked == [overview.content]


def test_general_answer_judges_sections_tied_to_no_retailer_first():
    # Measured: the Walmart fee section was closer to "the aco fee in general" than the overview.
    checked = []
    check_sections("fee in general", [FEE, IMAP], lambda q, p: checked.append(p), general=True)
    assert checked == [IMAP.content, FEE.content]


def test_sections_for_the_asked_retailer_or_none_are_judged():
    checked = []
    check_sections("walmart fee?", [FEE, IMAP, PKC_OTHER], lambda q, p: checked.append(p))
    assert checked == [FEE.content, IMAP.content]
    checked.clear()
    check_sections("what is the fee?", [FEE, PKC_OTHER], lambda q, p: checked.append(p))
    assert checked == [FEE.content, PKC_OTHER.content]


# --- The model's own wording ----------------------------------------------------------------

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


def ok(reply, hit, question="q", hits=None):
    """grounded() with the whole section body as the judge's quote."""
    return grounded(reply, Check(hit, body(hit), True), hits or [hit], question)


def test_faithful_rewording_holds_up():
    assert ok("For Walmart, most items cost $2.50 to $5 each.", FEE, "walmart fee?")
    assert ok("Unfortunately Vault and Venn cards aren't accepted.", CARDS)
    assert ok("Sure! Just open a ticket in #make-a-ticket.", LINKS)


def test_amount_or_number_the_docs_lack_is_rejected():
    assert not ok("For Walmart, most items cost $2.50 to $6.", FEE, "walmart fee?")
    assert not ok("For Walmart, most items cost $3 each.", FEE, "walmart fee?")
    assert not ok("For Walmart, it takes 3 days, at $2.50 to $5.", FEE, "walmart fee?")


def test_link_or_channel_the_docs_lack_is_rejected():
    assert not ok("Open a ticket in #support.", LINKS)
    assert not ok("Go to trevona.ca/signup to sign up.", LINKS)


def test_retailer_fee_passed_off_as_the_general_fee_is_rejected():
    # Reported: "The fee is $2.50 to $5 per item" for "what is the aco fee" (Walmart's fee).
    assert not ok("The ACO fee is $2.50 to $5 per item.", FEE, "what is the aco fee")
    assert ok("The ACO fee is $2.50 to $5 per item.", FEE, "what is the walmart fee")


def test_fee_said_for_the_wrong_retailer_is_rejected():
    assert not ok("On Costco Canada most items are $2.50 to $5.", FEE, "fee?")


def test_overview_amounts_keep_the_retailer_their_sentence_names():
    general = "how much is the aco fee in general"
    faithful = (
        "It depends on the retailer. On Pokémon Center Canada an Elite Trainer Box is $25 CAD, "
        "and Walmart Canada items are mostly $2.50 to $5."
    )
    assert ok(faithful, OVERVIEW, general)
    assert not ok("The ACO fee is $25 CAD per item.", OVERVIEW, general)
    assert not ok("On Walmart Canada it's $25 CAD.", OVERVIEW, general)


def test_retailer_carries_over_to_the_next_sentence():
    reply = "On Pokémon Center Canada an Elite Trainer Box is $25 CAD. Costco fees come later."
    assert ok(reply, OVERVIEW, "fees in general")
    assert not ok("Costco fees come later. An Elite Trainer Box is $25 CAD.", OVERVIEW, "fees?")


PKC_FEES = section(
    "How much is the ACO fee in general?",
    "On Pokémon Center Canada, pre-orders are $25 CAD for an Elite Trainer Box, $20 CAD for a "
    "Booster Box and $5 CAD for a Booster Bundle. On Walmart Canada, most items are $2.50 to $5.",
    0.2,
)


def test_product_fee_passed_off_as_every_pre_orders_fee_is_rejected():
    # Measured: "$25 CAD for pre-orders on Pokémon Center Canada" ($20 for a Booster Box).
    general = "how much is the aco fee in general"
    assert not ok("It's $25 CAD for pre-orders on Pokémon Center Canada.", PKC_FEES, general)
    faithful = "On Pokémon Center Canada it's $25 CAD for an ETB and $20 CAD for a Booster Box."
    assert ok(faithful, PKC_FEES, general)


def test_fees_swapped_within_one_sentence_are_rejected():
    general = "fees in general"
    swapped = "On Pokémon Center Canada it's $20 CAD for an ETB and $25 CAD for a Booster Box."
    assert not ok(swapped, PKC_FEES, general)
    shops = "Pokémon Center Canada charges $2.50 to $5, and Walmart Canada $25 CAD for an ETB."
    assert not ok(shops, PKC_FEES, general)


def test_amount_shared_by_two_fees_matches_either_but_not_a_third():
    # $5 is both the Booster Bundle fee and the top of Walmart's range.
    general = "fees in general"
    assert ok("A Booster Bundle on Pokémon Center Canada is $5 CAD.", PKC_FEES, general)
    assert ok("Walmart Canada items are mostly $2.50 to $5.", PKC_FEES, general)
    assert not ok("An ETB on Pokémon Center Canada is $5 CAD.", PKC_FEES, general)


def test_flipped_yes_or_no_is_rejected():
    assert not ok("Yes, Vault and Venn cards are accepted.", CARDS)
    assert not ok("Most Walmart items don't cost $2.50 to $5.", FEE, "walmart fee?")


def test_reply_without_the_documented_answer_is_rejected():
    assert not ok("Happy to help with that!", FEE, "walmart fee?")
    assert not ok("", FEE)


def test_verbatim_section_drops_a_bare_opening_yes_or_no():
    # "Can I get a refund if it arrives damaged?" got "Yes. Once an item is delivered, ...".
    assert verbatim(CARDS) == "Vault or Venn cards are not accepted."
    assert verbatim(FEE) == body(FEE)


BOX = section(
    "How much is the ACO fee for a Booster Box on Pokémon Center Canada?",
    "Trevona charges an ACO fee of $20 CAD for a Booster Box (pre-order) on Pokémon Center "
    "Canada, paid after delivery.",
    0.2,
)


def test_retailer_named_after_the_fee_in_the_same_sentence_counts():
    reply = "It's $20 CAD for a Booster Box (pre-order) on Pokémon Center Canada."
    assert ok(reply, BOX, "booster box pre-order fee?")


CANCELLED = section(
    "Do I pay the ACO fee if my order is cancelled?",
    "No. The ACO fee is only charged on orders that are delivered. If the retailer cancels an "
    "order, there is no ACO fee.",
    0.2,
)


def test_opening_yes_against_a_quoted_no_is_rejected_even_if_the_rest_is_copied():
    # Measured: the documented sentences followed, so the closest sentence alone agreed.
    copied = (
        "Yes, you still pay the ACO fee if your order got cancelled. The ACO fee is only charged "
        "on orders that are delivered."
    )
    assert not ok(copied, CANCELLED, "do I pay if it got cancelled?")
    assert ok("No, the ACO fee is only charged on orders that are delivered.", CANCELLED)


PRIME = section(
    "Do I need Amazon Prime?",
    "Amazon Prime is not required, but Prime accounts generally see higher success rates.",
    0.2,
)


def test_quote_mixing_a_no_and_a_yes_is_matched_clause_by_clause():
    reply = "No, you don't need Amazon Prime. However, Prime accounts generally see higher success."
    assert ok(reply, PRIME, "do I need prime?")
    assert not ok("Yes, Amazon Prime is required.", PRIME, "do I need prime?")


def test_reply_naming_a_retailer_the_question_does_not_ask_about_is_rejected():
    # Measured: "Can I jig my address for Costco?" got Walmart's guidance.
    walmart = section("Can I jig for Walmart Canada?", "No guidance was given for Walmart.", 0.2)
    assert not ok("No guidance was given for Walmart.", walmart, "can I jig for costco?")


SHIPMENT = section(
    "I received my shipment. What next?",
    "Reply in your support ticket or ACO fee ticket so the team can confirm what you owe.",
    0.2,
)


def test_invented_claims_are_rejected_but_conversational_wording_is_not():
    # Measured: the model invented a "My Orders" section to check.
    invented = (
        "Reply in your support ticket so the team can confirm what you owe. Please check your "
        "order details in the My Orders section of your account to confirm quantities."
    )
    assert not ok(invented, SHIPMENT, "I got my shipment")
    friendly = "Great! Just reply in your support ticket so the team can confirm what you owe."
    assert ok(friendly, SHIPMENT, "I got my shipment")
