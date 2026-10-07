from rag_engine.assistant.verify import body, check_sections, find_answer, is_supported
from rag_engine.models import Hit


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
