"""Tesseract OCR that keeps every word's bounding box and confidence."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class Word:
    text: str
    left: int
    top: int
    width: int
    height: int
    conf: float
    line: int = 0  # reading-order line index, set by sort_reading_order

    @property
    def box(self) -> tuple[int, int, int, int]:
        return self.left, self.top, self.left + self.width, self.top + self.height


def read_words(image: np.ndarray, lang: str = "eng+deu", min_conf: float = 30.0) -> list[Word]:
    """Run Tesseract and return words in reading order.

    ``lang`` defaults to English + German because the original client's cards
    were German; any installed Tesseract language pack works.
    """
    import pytesseract

    data = pytesseract.image_to_data(image, lang=lang, config="--psm 11", output_type=pytesseract.Output.DICT)
    words = []
    for i, text in enumerate(data["text"]):
        text = (text or "").strip()
        conf = float(data["conf"][i])
        if not text or conf < min_conf or not any(ch.isalnum() for ch in text):
            continue
        words.append(Word(text, data["left"][i], data["top"][i], data["width"][i], data["height"][i], conf))
    return sort_reading_order(words)


def sort_reading_order(words: list[Word]) -> list[Word]:
    """Group words into lines by vertical overlap, then read left to right.

    Sparse-text mode (--psm 11) finds scattered card text well but returns it in
    no useful order; NER needs neighbours like "John" "Smith" next to each other.
    """
    lines: list[list[Word]] = []
    for w in sorted(words, key=lambda w: (w.top, w.left)):
        mid = w.top + w.height / 2
        for line in lines:
            ref = line[0]
            if ref.top <= mid <= ref.top + ref.height:
                line.append(w)
                break
        else:
            lines.append([w])
    return [
        Word(w.text, w.left, w.top, w.width, w.height, w.conf, n)
        for n, line in enumerate(lines)
        for w in sorted(line, key=lambda w: w.left)
    ]
