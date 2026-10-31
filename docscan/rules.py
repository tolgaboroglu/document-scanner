"""Rule-based extractors for fields with a fixed shape: e-mail, website, phone.

The pipeline uses them as a fallback for spans the NER model missed, and to
normalise values (digits-only phone numbers, lower-case e-mail and web).
"""

from __future__ import annotations

import re
from typing import Sequence

EMAIL = re.compile(r"^[\w.+-]+@[\w-]+(\.[\w-]+)*\.[a-z]{2,}$", re.IGNORECASE)
WEB = re.compile(r"^(https?://)?(www\.)?[\w-]+(\.[\w-]+)*\.(com|net|org|de|in|co|io|info|biz|eu|uk|tr|edu|gov|ai|app|me)(/\S*)?$", re.IGNORECASE)
PHONE_CHARS = re.compile(r"^[+()\d\-./]+,?$")
STRIP = "\"'“”‘’,;:|()[]{}<>"


def _clean(word: str) -> str:
    return word.strip(STRIP)


def _looks_like_phone_part(word: str) -> bool:
    w = word.strip(STRIP.replace("(", "").replace(")", ""))
    return bool(w) and bool(PHONE_CHARS.match(w)) and sum(c.isdigit() for c in w) >= 2


def find_spans(words: Sequence[str]) -> list[tuple[int, int, str]]:
    """Return (start, end, label) word spans for EMAIL, WEB and PHONE."""
    spans: list[tuple[int, int, str]] = []
    i, n = 0, len(words)
    while i < n:
        w = _clean(words[i])
        if "@" in w and EMAIL.match(w):
            spans.append((i, i + 1, "EMAIL"))
            i += 1
            continue
        if "@" not in w and WEB.match(w):
            spans.append((i, i + 1, "WEB"))
            i += 1
            continue
        if _looks_like_phone_part(words[i]):
            j, digits = i, 0
            while j < n and _looks_like_phone_part(words[j]) and digits < 15:
                digits += sum(c.isdigit() for c in words[j])
                j += 1
                if words[j - 1].rstrip().endswith(","):  # "8881," ends this number
                    break
            if 7 <= digits <= 15:
                spans.append((i, j, "PHONE"))
                i = j
                continue
        i += 1
    return spans


def normalise_phone(text: str) -> str:
    keep_plus = text.strip().startswith("+")
    digits = re.sub(r"\D", "", text)
    return ("+" if keep_plus else "") + digits


def normalise_web(text: str) -> str:
    return _clean(text).lower()


def normalise_email(text: str) -> str:
    return _clean(text).lower().lstrip(".-_")
