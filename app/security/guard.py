"""Cheap, deterministic input guardrails that run before any LLM call.

This is a first line of defence, not the only one: retrieved content is also fenced as
untrusted data in the system prompt, and the agent has no tools that can move money or
read customer data.
"""

import re
import unicodedata
from dataclasses import dataclass

_INJECTION_PATTERNS: list[tuple[str, re.Pattern[str]]] = [
    (
        "override_instructions",
        re.compile(
            r"\b(ignore|disregard|forget)\b.{0,30}\b(previous|above|prior|all|system)\b"
            r".{0,20}\b(instructions?|prompts?|rules?)\b"
        ),
    ),
    (
        "override_instructions",
        re.compile(r"\b(bo qua|phot lo|quen)\b.{0,30}\b(huong dan|chi dan|chi thi|quy tac|lenh)\b"),
    ),
    (
        "prompt_exfiltration",
        re.compile(
            r"\b(reveal|show|print|repeat|leak)\b.{0,30}\b(system prompt|instructions|prompt)\b"
        ),
    ),
    (
        "prompt_exfiltration",
        re.compile(
            r"\b(tiet lo|in ra|hien thi|cho (toi|tao) xem)\b.{0,30}\b(prompt|chi dan he thong)\b"
        ),
    ),
    (
        "role_hijack",
        re.compile(r"\b(you are now|act as|pretend to be|developer mode|jailbreak|dan mode)\b"),
    ),
    ("role_hijack", re.compile(r"\b(tu bay gio ban la|hay dong vai)\b")),
]


def _normalize(text: str) -> str:
    """Lower-case and strip Vietnamese diacritics so 'Bỏ qua' == 'bo qua'."""
    text = text.lower().replace("đ", "d")
    text = unicodedata.normalize("NFD", text)
    return "".join(c for c in text if unicodedata.category(c) != "Mn")


@dataclass(frozen=True)
class GuardResult:
    allowed: bool
    reason: str | None = None


def check_input(text: str, max_chars: int) -> GuardResult:
    if not text.strip():
        return GuardResult(False, "empty")
    if len(text) > max_chars:
        return GuardResult(False, "too_long")
    norm = _normalize(text)
    for reason, pattern in _INJECTION_PATTERNS:
        if pattern.search(norm):
            return GuardResult(False, reason)
    return GuardResult(True)


REFUSAL_MESSAGE = (
    "Xin lỗi, tôi không thể xử lý yêu cầu này. Tôi chỉ hỗ trợ giải đáp thông tin từ tài liệu "
    "sản phẩm và chính sách của ngân hàng."
)
