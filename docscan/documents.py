"""Invoices and old paper records: the document types the original client needed digitised.

Invoices: read the page and pull out the bookkeeping fields (number, date, net,
VAT, total, VAT ID, IBAN), then check that they add up.

Old records: remove yellowing and stains, then produce plain text and a
searchable PDF (image with an invisible text layer) for the archive.
"""

from __future__ import annotations

import re
from dataclasses import asdict, dataclass, field
from datetime import date
from decimal import Decimal, InvalidOperation

import cv2
import numpy as np

# ---------------------------------------------------------------- image + OCR


def clean_paper(image: np.ndarray, block: int = 51) -> np.ndarray:
    """Flatten the paper background: divide by a heavily blurred copy of the page.

    Stains, yellowing and uneven light are large, smooth shapes; letters are small
    and sharp. Dividing by a median-blurred background keeps the letters and turns
    the rest white. On the synthetic 1978 letter this takes Tesseract from 10 to
    14 of 14 lines read exactly.
    """
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if image.ndim == 3 else image
    background = cv2.medianBlur(gray, block)
    return cv2.divide(gray, background, scale=255)


def read_text(image: np.ndarray, lang: str = "deu+eng", psm: int = 4) -> str:
    import pytesseract

    return pytesseract.image_to_string(image, lang=lang, config=f"--psm {psm}")


def searchable_pdf(image: np.ndarray, lang: str = "deu+eng") -> bytes:
    """A PDF that shows the scan and has selectable, searchable text underneath."""
    import pytesseract

    return pytesseract.image_to_pdf_or_hocr(image, extension="pdf", lang=lang, config="--psm 4")


# ------------------------------------------------------------------- invoices

AMOUNT = r"(-?\d{1,3}(?:[.,\s]\d{3})*[.,]\d{2})"
LABELS = {
    "invoice_number": r"(?:Rechnungs-?\s?(?:nummer|nr\.?)|Rechnung\s+Nr\.?|Invoice\s+(?:No\.?|Number|#))\s*[:#]?\s*([A-Z0-9][A-Z0-9\-/.]{2,})",
    "date": r"(?:Rechnungsdatum|Belegdatum|Invoice\s+date|Datum|Date)\s*[:]?\s*(\d{1,2}\.\d{1,2}\.\d{2,4}|\d{4}-\d{2}-\d{2})",
    "net": rf"(?:Nettobetrag|Netto|Zwischensumme|Subtotal|Net\s+amount)\b[^\n\d]*{AMOUNT}",
    "vat": rf"(?:MwSt|USt|Umsatzsteuer|VAT)\b[^\n]*?{AMOUNT}\s*€?\s*$",
    "total": rf"(?:Gesamtbetrag|Gesamtsumme|Rechnungsbetrag|Endbetrag|Bruttobetrag|Total\s+due|Total)\b[^\n\d]*{AMOUNT}",
    "vat_id": r"(?:USt-?\s?Id(?:\.?-?Nr)?\.?|VAT\s+(?:ID|No\.?)|UID)\s*[:.]?\s*([A-Z]{2}\s?[0-9A-Z]{8,12})\b",
    "iban": r"IBAN\s*[:.]?\s*([A-Z]{2}\d{2}(?:\s?[A-Z0-9]{4}){2,7}(?:\s?[A-Z0-9]{1,3})?)",
}
VAT_RATE = re.compile(r"(\d{1,2}(?:[.,]\d)?)\s*%")


def parse_amount(text: str) -> Decimal | None:
    """Parse '1.234,56', '1,234.56', '817,53' or '817.53' into a Decimal."""
    s = re.sub(r"[\s€]", "", text)
    if not s:
        return None
    if "," in s and "." in s:
        s = s.replace(".", "").replace(",", ".") if s.rfind(",") > s.rfind(".") else s.replace(",", "")
    elif "," in s:
        s = s.replace(",", ".")
    try:
        return Decimal(s)
    except InvalidOperation:
        return None


def iban_is_valid(iban: str) -> bool:
    """ISO 13616 mod-97 check: catches almost every OCR misread digit."""
    s = re.sub(r"\s", "", iban).upper()
    if not re.fullmatch(r"[A-Z]{2}\d{2}[A-Z0-9]{10,30}", s):
        return False
    digits = "".join(str(int(c, 36)) for c in s[4:] + s[:4])
    return int(digits) % 97 == 1


def parse_date(text: str) -> str | None:
    m = re.fullmatch(r"(\d{1,2})\.(\d{1,2})\.(\d{2,4})", text)
    try:
        if m:
            d, mo, y = (int(x) for x in m.groups())
            return date(y + 2000 if y < 100 else y, mo, d).isoformat()
        return date.fromisoformat(text).isoformat()
    except ValueError:
        return None


@dataclass
class Invoice:
    seller: str | None = None
    invoice_number: str | None = None
    date: str | None = None          # ISO 8601
    net: str | None = None           # amounts as plain decimals, e.g. "687.00"
    vat_rate: str | None = None      # percent, e.g. "19"
    vat: str | None = None
    total: str | None = None
    currency: str = "EUR"
    vat_id: str | None = None
    iban: str | None = None
    checks: list[dict] = field(default_factory=list)

    def to_dict(self) -> dict:
        return asdict(self)


def _search(key: str, text: str) -> str | None:
    m = re.search(LABELS[key], text, re.IGNORECASE | re.MULTILINE)
    return m.group(1).strip() if m else None


def parse_invoice(text: str) -> Invoice:
    lines = [ln.strip() for ln in text.splitlines() if ln.strip()]
    inv = Invoice(seller=lines[0] if lines else None)
    inv.invoice_number = _search("invoice_number", text)
    raw_date = _search("date", text)
    inv.date = parse_date(raw_date) if raw_date else None

    amounts = {}
    for key in ("net", "vat", "total"):
        raw = _search(key, text)
        value = parse_amount(raw) if raw else None
        amounts[key] = value
        setattr(inv, key, f"{value:.2f}" if value is not None else None)
    vat_line = next((ln for ln in lines if re.search(r"\b(MwSt|USt|Umsatzsteuer|VAT)\b", ln, re.I) and "%" in ln), "")
    rate = VAT_RATE.search(vat_line)
    inv.vat_rate = rate.group(1).replace(",", ".") if rate else None

    vat_id = _search("vat_id", text)
    inv.vat_id = re.sub(r"\s", "", vat_id) if vat_id else None
    iban = _search("iban", text)
    inv.iban = " ".join(re.findall(r".{1,4}", re.sub(r"\s", "", iban))) if iban else None

    inv.checks = run_checks(inv, amounts)
    return inv


def run_checks(inv: Invoice, amounts: dict) -> list[dict]:
    """Bookkeeping sanity checks that flag OCR errors before data reaches accounting."""
    checks = []
    net, vat, total = amounts.get("net"), amounts.get("vat"), amounts.get("total")
    if None not in (net, vat, total):
        ok = abs(net + vat - total) <= Decimal("0.01")
        checks.append({"name": "Net + VAT = total", "ok": ok,
                       "detail": f"{net:.2f} + {vat:.2f} = {net + vat:.2f} (invoice says {total:.2f})"})
    if net is not None and vat is not None and inv.vat_rate:
        expected = (net * Decimal(inv.vat_rate) / 100).quantize(Decimal("0.01"))
        checks.append({"name": f"VAT is {inv.vat_rate} % of net", "ok": abs(expected - vat) <= Decimal("0.01"),
                       "detail": f"expected {expected:.2f}, found {vat:.2f}"})
    if inv.iban:
        ok = iban_is_valid(inv.iban)
        checks.append({"name": "IBAN checksum", "ok": ok, "detail": "valid" if ok else "check digits do not match"})
    for key, label in (("invoice_number", "Invoice number"), ("date", "Invoice date"), ("total", "Total amount")):
        if getattr(inv, key) is None:
            checks.append({"name": f"{label} found", "ok": False, "detail": "not found, enter it manually"})
    return checks
