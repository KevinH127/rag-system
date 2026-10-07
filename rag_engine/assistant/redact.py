import re
from dataclasses import dataclass

PLACEHOLDER = "[REDACTED]"

_CARD = re.compile(r"(?<!\d)(?:\d[ -]?){12,18}\d(?!\d)")
_OTP = re.compile(
    r"(?i)\b(otp|one[- ]time (?:pass)?code|passcode|verification code|security code|code)\b"
    r"([^\d\n]{0,25})(\d{4,8})(?!\d)"
)
_PASSWORD = re.compile(r"(?i)\b(password|passwd|pwd|pw)\b(\s*(?:is|was|:|=)\s*)(\S+)")


@dataclass(frozen=True)
class Redacted:
    text: str
    kinds: tuple[str, ...]


def _luhn(digits: str) -> bool:
    total = 0
    for i, ch in enumerate(reversed(digits)):
        n = int(ch)
        if i % 2:
            n = n * 2 - 9 if n > 4 else n * 2
        total += n
    return total % 10 == 0


def redact(text: str) -> Redacted:
    """Replace card numbers, one-time codes and passwords so they never reach the LLM or logs."""
    kinds: list[str] = []

    def card(m: re.Match[str]) -> str:
        digits = re.sub(r"\D", "", m.group())
        if 13 <= len(digits) <= 19 and _luhn(digits):
            kinds.append("card number")
            return PLACEHOLDER
        return m.group()

    text = _CARD.sub(card, text)

    def otp(m: re.Match[str]) -> str:
        kinds.append("one-time code")
        return f"{m.group(1)}{m.group(2)}{PLACEHOLDER}"

    text = _OTP.sub(otp, text)

    def password(m: re.Match[str]) -> str:
        kinds.append("password")
        return f"{m.group(1)}{m.group(2)}{PLACEHOLDER}"

    text = _PASSWORD.sub(password, text)
    return Redacted(text, tuple(dict.fromkeys(kinds)))


_CLAUSE = re.compile(r"[^,.;!?\n]+[,.;!?]?\s*")


def drop_secret_clauses(text: str) -> str:
    """Remove the clauses that contained a redacted secret, keeping the rest of the message."""
    kept = [c for c in _CLAUSE.findall(text) if PLACEHOLDER not in c]
    return "".join(kept).strip()
