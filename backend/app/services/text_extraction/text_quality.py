import re

_CZECH_CHARS = set("áčďéěíňóřšťúůýžÁČĎÉĚÍŇÓŘŠŤÚŮÝŽ")
_CZECH_WORDS = re.compile(
    r"\b(faktura|dodavatel|odběratel|odberatel|dph|ičo|ico|dič|dic|datum|celkem|částka|castka)\b",
    re.IGNORECASE,
)


def has_usable_text(
    text: str,
    *,
    min_chars: int,
    max_garbage_ratio: float,
    require_czech_signal: bool = True,
) -> bool:
    """Heuristic for whether extracted text is good enough to skip OCR."""
    stripped = (text or "").strip()
    if len(stripped) < min_chars:
        return False

    meaningful = sum(1 for char in stripped if char.isalnum() or char.isspace())
    garbage_ratio = 1 - (meaningful / max(len(stripped), 1))
    if garbage_ratio > max_garbage_ratio:
        return False

    if not require_czech_signal:
        return True

    has_czech_chars = any(char in _CZECH_CHARS for char in stripped)
    has_czech_words = bool(_CZECH_WORDS.search(stripped))
    return has_czech_chars or has_czech_words
