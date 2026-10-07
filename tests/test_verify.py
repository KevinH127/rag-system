from helpers import section

from rag_engine.assistant.verify import body, check_sections, is_supported, verbatim, verified

IMAP = section("Why is IMAP required?", "IMAP must be set up correctly or checkouts fail.", 0.3)
FEE = section("What is the Walmart fee?", "Most Walmart items cost $2.50 to $5 per item.", 0.2)
CARDS = section("Can I use a Vault card?", "No. Vault or Venn cards are not accepted.", 0.2)
PKC_OTHER = section(
    "How much is the ACO fee for other Pokémon Center Canada items?",
    "The ACO fee for other Pokémon Center Canada items is set after the drop.",
    0.25,
)


def quotes_body(question, passage):
    return passage.partition("\n\n")[2]


def answer(question, hits, judge):
    check = verified(check_sections(question, hits, judge))
    return check.hit if check else None


def judged(question, hits, **kwargs):
    """The section contents the judge is shown, in order."""
    shown = []
    check_sections(question, hits, lambda q, p: shown.append(p), **kwargs)
    return shown


# --- Section text ---------------------------------------------------------------------------


def test_body_drops_the_label_line():
    assert body(FEE) == "Most Walmart items cost $2.50 to $5 per item."


def test_verbatim_section_drops_a_bare_opening_yes_or_no():
    # It answers the section's heading, which the customer never sees.
    assert verbatim(CARDS) == "Vault or Venn cards are not accepted."
    assert verbatim(FEE) == body(FEE)


# --- Quotes ---------------------------------------------------------------------------------


def test_quote_matches_despite_case_punctuation_and_spacing():
    assert is_supported("most walmart items cost $2.50  to $5", body(FEE))


def test_invented_or_too_short_quotes_are_rejected():
    assert not is_supported("Walmart items are free", body(FEE))
    assert not is_supported("Most", body(FEE))
    assert not is_supported("", body(FEE))


def test_quoting_the_heading_does_not_count_as_an_answer():
    assert answer("walmart fee?", [FEE], lambda q, p: "What is the Walmart fee?") is None


# --- Which sections are judged, in what order -----------------------------------------------


def test_first_verified_section_in_distance_order_wins():
    assert answer("q", [IMAP, FEE], quotes_body) == FEE  # FEE is closer


def test_falls_through_to_the_next_section_when_the_closest_has_no_answer():
    def judge(question, passage):
        return "IMAP must be set up correctly" if "IMAP" in passage else None

    assert answer("q", [IMAP, FEE], judge) == IMAP


def test_only_the_closest_few_sections_are_checked(monkeypatch):
    from rag_engine.config import settings

    monkeypatch.setattr(settings, "verify_top_n", 1)
    assert judged("q", [IMAP, FEE]) == [FEE.content]


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
    assert [c.hit for c in check_sections("q", [IMAP, FEE], quotes_body)] == [FEE]


def test_section_about_another_retailer_is_never_judged():
    assert judged("ACO fee for Bandai drops?", [PKC_OTHER, FEE]) == []


def test_general_section_whose_text_covers_only_other_retailers_is_never_judged():
    overview = section(
        "How much is the ACO fee in general?",
        "Walmart Canada is $2.50 to $5. Costco is later.",
        0.2,
    )
    assert judged("aco fee bandai", [overview]) == []
    assert judged("aco fee costco", [overview]) == [overview.content]


def test_sections_for_the_asked_retailer_or_none_are_judged():
    assert judged("walmart fee?", [FEE, IMAP, PKC_OTHER]) == [FEE.content, IMAP.content]
    assert judged("what is the fee?", [FEE, PKC_OTHER]) == [FEE.content, PKC_OTHER.content]


def test_general_answer_judges_sections_tied_to_no_retailer_first():
    assert judged("fee in general", [FEE, IMAP], general=True) == [IMAP.content, FEE.content]
