"""Combine NER and rules into clean contact fields with bounding boxes."""

from __future__ import annotations

import re
from dataclasses import asdict, dataclass, field
from typing import Sequence

from .ner import LABELS
from .ocr import Word
from .rules import find_spans, normalise_email, normalise_phone, normalise_web


@dataclass
class Field:
    label: str
    value: str
    box: tuple[int, int, int, int] | None = None
    source: str = "ner"  # "ner", "rules", "layout" or "llm"


@dataclass
class Contact:
    fields: list[Field] = field(default_factory=list)

    def values(self, label: str) -> list[str]:
        return [f.value for f in self.fields if f.label == label]

    def to_dict(self) -> dict:
        grouped = {lab: self.values(lab) for lab in LABELS}
        return {"fields": [asdict(f) for f in self.fields], "grouped": grouped}


def merge_spans(ner: Sequence[tuple[int, int, str]], rules: Sequence[tuple[int, int, str]]):
    """NER first; rules fill in phone/e-mail/web spans the model missed.

    Measured on the held-out cards (docs/ner_metrics.json): letting rules override
    the model lowered e-mail and web F1, because OCR noise breaks strict patterns
    more often than it confuses the model. Used as a fallback, rules only add
    spans that do not overlap anything the model found.
    """
    out = [(s, e, lab, "ner") for s, e, lab in ner]
    taken = {i for s, e, _ in ner for i in range(s, e)}
    for s, e, lab in rules:
        if not taken.intersection(range(s, e)):
            out.append((s, e, lab, "rules"))
            taken.update(range(s, e))
    return sorted(out)


def split_at_lines(spans, lines: Sequence[int]):
    """Cut spans at line breaks; keep the longest piece (entities rarely wrap on cards)."""
    out = []
    for s, e, lab, src in spans:
        pieces, start = [], s
        for i in range(s + 1, e):
            if lines[i] != lines[i - 1]:
                pieces.append((start, i))
                start = i
        pieces.append((start, e))
        a, b = max(pieces, key=lambda p: p[1] - p[0])
        out.append((a, b, lab, src))
    return out


def validate(spans, tokens: Sequence[str]):
    """Fix labels the model gets wrong on structured fields.

    A single word that matches the e-mail pattern is an e-mail even if the model
    called it a website (and vice versa); NAME/ORG/DES spans that are really an
    e-mail, URL or phone number are relabelled the same way.
    """
    rule_label = {(s, e): lab for s, e, lab in find_spans(tokens)}
    out = []
    for s, e, lab, src in spans:
        fixed = rule_label.get((s, e))
        if fixed and fixed != lab:
            out.append((s, e, fixed, "rules"))
        elif lab == "EMAIL" and not any("@" in t for t in tokens[s:e]):
            continue
        else:
            out.append((s, e, lab, src))
    return out


def guess_name_from_layout(words: Sequence[Word], spans):
    """If no NAME was found, take the tallest unlabelled line of 2–3 capitalised words.

    On business cards the person's name is usually the biggest text.
    """
    taken = {i for s, e, *_ in spans for i in range(s, e)}
    lines: dict[int, list[int]] = {}
    for i, w in enumerate(words):
        lines.setdefault(w.line, []).append(i)
    best, best_h = None, 0.0
    for idx in lines.values():
        if taken.intersection(idx) or not 2 <= len(idx) <= 3:
            continue
        texts = [words[i].text for i in idx]
        if not all(t[:1].isupper() and t.replace("-", "").isalpha() and not t.isupper() for t in texts):
            continue
        h = sum(words[i].height for i in idx) / len(idx)
        if h > best_h:
            best, best_h = (idx[0], idx[-1] + 1, "NAME", "layout"), h
    return best


def clean_value(label: str, text: str) -> str:
    if label == "PHONE":
        return normalise_phone(text)
    if label == "EMAIL":
        return normalise_email(text)
    if label == "WEB":
        return normalise_web(text)
    text = re.sub(r"[^\w&.,'\- ]", "", text).strip(" .,-")
    return text.title() if text.isupper() or text.islower() else text


def build_contact(words: Sequence[Word], ner_spans, use_rules: bool = True) -> Contact:
    tokens = [w.text for w in words]
    spans = merge_spans(ner_spans, find_spans(tokens) if use_rules else [])
    if use_rules:
        spans = validate(split_at_lines(spans, [w.line for w in words]), tokens)
        if not any(lab == "NAME" for _, _, lab, _ in spans):
            guess = guess_name_from_layout(words, spans)
            if guess:
                spans = sorted(spans + [guess])
    contact = Contact()
    seen = set()
    for s, e, lab, source in spans:
        group = words[s:e]
        value = clean_value(lab, " ".join(w.text for w in group))
        if not value or (lab, value) in seen:
            continue
        seen.add((lab, value))
        box = (min(w.left for w in group), min(w.top for w in group),
               max(w.left + w.width for w in group), max(w.top + w.height for w in group))
        contact.fields.append(Field(lab, value, box, source))
    return contact
