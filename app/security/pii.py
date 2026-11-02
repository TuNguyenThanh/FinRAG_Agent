"""Deterministic PII masking for Vietnamese banking text.

Runs at the API boundary so that the LLM provider, the semantic cache, logs and the
audit trail only ever see masked text.
"""

import re
from dataclasses import dataclass, field

_CARD = re.compile(r"\b(?:\d[ -]?){12,18}\d\b")
_CCCD = re.compile(r"\b0\d{11}\b")  # 12-digit citizen ID (CCCD)
_PHONE = re.compile(r"(?<!\d)(?:\+?84|0)(?:3|5|7|8|9)\d{8}(?!\d)")
_EMAIL = re.compile(r"\b[\w.+-]+@[\w-]+\.[\w.-]+\b")
# Account numbers look like any other number, so only mask digits that follow a cue
# word. This avoids masking amounts such as "50000000 VND".
_ACCOUNT = re.compile(
    r"(?i)(\b(?:stk|số\s*tài\s*khoản|số\s*tk|tài\s*khoản|tk|account(?:\s*(?:no\.?|number))?)"
    r"\s*[:#]?\s*)(\d{6,19})\b"
)


def _luhn_ok(digits: str) -> bool:
    total, parity = 0, len(digits) % 2
    for i, ch in enumerate(digits):
        d = int(ch)
        if i % 2 == parity:
            d *= 2
            if d > 9:
                d -= 9
        total += d
    return total % 10 == 0


@dataclass
class MaskResult:
    text: str
    found: dict[str, int] = field(default_factory=dict)

    @property
    def has_pii(self) -> bool:
        return bool(self.found)


def mask_pii(text: str) -> MaskResult:
    found: dict[str, int] = {}

    def bump(kind: str) -> None:
        found[kind] = found.get(kind, 0) + 1

    def card_sub(m: re.Match[str]) -> str:
        digits = re.sub(r"\D", "", m.group(0))
        if 13 <= len(digits) <= 19 and _luhn_ok(digits):
            bump("card")
            return "[CARD]"
        return m.group(0)

    def simple(kind: str, token: str):
        def _sub(_: re.Match[str]) -> str:
            bump(kind)
            return token

        return _sub

    def account_sub(m: re.Match[str]) -> str:
        bump("account")
        return f"{m.group(1)}[ACCOUNT]"

    # Cue-word account numbers first: an explicit "STK ..." label beats pattern guessing.
    text = _ACCOUNT.sub(account_sub, text)
    text = _CARD.sub(card_sub, text)
    text = _CCCD.sub(simple("cccd", "[CCCD]"), text)
    text = _PHONE.sub(simple("phone", "[PHONE]"), text)
    text = _EMAIL.sub(simple("email", "[EMAIL]"), text)
    return MaskResult(text=text, found=found)
