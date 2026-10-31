"""End-to-end pipelines for the three document types.

card     photo → flattened card → OCR words → contact fields (vCard)
invoice  photo → flattened page → OCR text → bookkeeping fields + checks
archive  photo → flattened, cleaned page → text + searchable PDF
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

from . import documents, scanner
from .documents import Invoice
from .extract import Contact, build_contact
from .ocr import Word, read_words

ROOT = Path(__file__).resolve().parent.parent
DEFAULT_MODEL = ROOT / "models" / "card_ner"


@dataclass
class ScanResult:
    card: np.ndarray            # flattened, enhanced card image
    words: list[Word]
    contact: Contact
    annotated: np.ndarray       # card with labelled boxes drawn


COLORS = {  # BGR, one per label
    "NAME": (64, 64, 220), "ORG": (40, 140, 40), "DES": (180, 110, 20),
    "PHONE": (30, 120, 200), "EMAIL": (150, 60, 150), "WEB": (120, 120, 0),
}


def annotate(card: np.ndarray, contact: Contact) -> np.ndarray:
    out = card.copy()
    for f in contact.fields:
        if not f.box:
            continue
        l, t, r, b = f.box
        color = COLORS.get(f.label, (0, 0, 0))
        cv2.rectangle(out, (l - 3, t - 3), (r + 3, b + 3), color, 2)
        cv2.putText(out, f.label, (l - 3, max(12, t - 7)), cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 1, cv2.LINE_AA)
    return out


@dataclass
class InvoiceResult:
    page: np.ndarray
    text: str
    invoice: Invoice


@dataclass
class ArchiveResult:
    page: np.ndarray            # cleaned page (background removed)
    text: str
    pdf: bytes                  # searchable PDF


class DocScanner:
    def __init__(self, model_dir: str | Path | None = None, extractor: str | None = None, lang: str | None = None):
        self.extractor_kind = extractor or os.getenv("EXTRACTOR", "ner")  # "ner" or "llm"
        self.lang = lang or os.getenv("OCR_LANG", "eng+deu")
        self.doc_lang = os.getenv("DOC_OCR_LANG", "deu+eng")
        self.ner = None
        self.llm = None
        if self.extractor_kind == "llm":
            from .llm import ClaudeExtractor

            self.llm = ClaudeExtractor()
        else:
            from .ner import CardNER

            self.ner = CardNER(model_dir or os.getenv("NER_MODEL", str(DEFAULT_MODEL)))

    @property
    def name(self) -> str:
        return self.llm.name if self.llm else f"{self.ner.name} + rules"

    def flatten(self, image: np.ndarray, corners: np.ndarray | None) -> np.ndarray:
        if corners is None:
            corners = scanner.detect_corners(image)
        card = scanner.warp(image, corners) if corners is not None else image
        return scanner.enhance(card)

    def extract(self, card: np.ndarray) -> ScanResult:
        words = read_words(card, lang=self.lang)
        if self.llm:
            contact = self.llm.extract(card)
        else:
            contact = build_contact(words, self.ner.predict([w.text for w in words]))
        return ScanResult(card, words, contact, annotate(card, contact))

    def scan(self, image: np.ndarray, corners: np.ndarray | None = None) -> ScanResult:
        """Business card → contact."""
        return self.extract(self.flatten(image, corners))

    def invoice(self, image: np.ndarray, corners: np.ndarray | None = None) -> InvoiceResult:
        page = self.flatten(image, corners)
        text = documents.read_text(page, lang=self.doc_lang)
        return InvoiceResult(page, text, documents.parse_invoice(text))

    def archive(self, image: np.ndarray, corners: np.ndarray | None = None) -> ArchiveResult:
        if corners is None:
            corners = scanner.detect_corners(image)
        page = scanner.warp(image, corners) if corners is not None else image
        clean = documents.clean_paper(page)
        return ArchiveResult(clean, documents.read_text(clean, lang=self.doc_lang), documents.searchable_pdf(clean, lang=self.doc_lang))


CardScanner = DocScanner  # backwards-compatible name
