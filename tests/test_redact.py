import pytest

from rag_engine.assistant.redact import PLACEHOLDER, drop_secret_clauses, redact


def test_valid_card_is_redacted():
    r = redact("my card is 4242 4242 4242 4242 please charge it")
    assert "4242" not in r.text and PLACEHOLDER in r.text
    assert r.kinds == ("card number",)


def test_invalid_luhn_number_is_left_alone():
    assert redact("order 1234 5678 9012 3456").kinds == ()


@pytest.mark.parametrize(
    "msg", ["my otp is 483920", "the verification code: 123456", "passcode 98765432"]
)
def test_otp_is_redacted(msg):
    r = redact(msg)
    assert r.kinds == ("one-time code",)
    assert not any(ch.isdigit() for ch in r.text)


def test_password_is_redacted():
    r = redact("my password is hunter2!")
    assert "hunter2" not in r.text and r.kinds == ("password",)


def test_plain_message_is_unchanged():
    msg = "How much is the fee for 2 booster boxes?"
    assert redact(msg).text == msg and redact(msg).kinds == ()


def test_password_followed_by_a_question_keeps_the_question():
    r = redact("my password is hunter22, why did checkout fail?")
    assert "hunter22" not in r.text
    assert drop_secret_clauses(r.text) == "why did checkout fail?"


def test_drop_secret_clauses_keeps_the_real_question():
    r = redact("my password is hunter2 and my otp is 482913, why did checkout fail?")
    assert drop_secret_clauses(r.text) == "why did checkout fail?"


def test_message_that_is_only_a_secret_becomes_empty():
    assert drop_secret_clauses(redact("my otp is 482913").text) == ""
