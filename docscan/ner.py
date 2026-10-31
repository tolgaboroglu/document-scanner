"""spaCy NER over OCR words.

The model runs on a Doc built directly from the OCR words, so every predicted
entity maps back to the exact words (and bounding boxes) it came from. v1
re-tokenised the joined text and matched tokens back by character offsets,
which broke whenever spaCy split a word differently from Tesseract.
"""

from __future__ import annotations

from pathlib import Path
from typing import Sequence

LABELS = ("NAME", "ORG", "DES", "PHONE", "EMAIL", "WEB")


def bio_to_entities(tags: Sequence[str]) -> list[tuple[int, int, str]]:
    """Turn BIO tags into (start, end, label) word spans.

    An I- tag that does not continue an entity of the same label starts a new one,
    so slightly malformed sequences still decode sensibly.
    """
    spans: list[tuple[int, int, str]] = []
    start, label = None, None
    for i, tag in enumerate(list(tags) + ["O"]):
        prefix, _, lab = tag.partition("-")
        continues = prefix == "I" and lab == label and start is not None
        if not continues and start is not None:
            spans.append((start, i, label))
            start, label = None, None
        if prefix in ("B", "I") and not continues:
            start, label = i, lab
    return spans


class CardNER:
    def __init__(self, model_dir: str | Path):
        import spacy

        self.nlp = spacy.load(model_dir)
        self.name = f"spaCy NER ({Path(model_dir).name})"

    def predict(self, words: Sequence[str]) -> list[tuple[int, int, str]]:
        """Return (start_word, end_word, label) spans for a list of OCR words."""
        from spacy.tokens import Doc

        if not words:
            return []
        doc = self.nlp(Doc(self.nlp.vocab, words=list(words)))
        return [(e.start, e.end, e.label_) for e in doc.ents if e.label_ in LABELS]
