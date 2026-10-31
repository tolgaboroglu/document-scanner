"""Generate synthetic document photos for demos and tests.

Three document types, matching what the original client needed digitised:
business cards, supplier invoices and old paper records. All people,
companies, numbers and domains are invented (domains use the reserved
example.com; the IBAN is the standard documentation example), so the
repository ships no real personal or client data.

    python scripts/make_samples.py            # writes samples/*.jpg and samples/*.json
"""

from __future__ import annotations

import json
import random
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

ROOT = Path(__file__).resolve().parent.parent
FONT_DIR = Path("/usr/share/fonts/truetype/dejavu")

CARDS = [
    {"NAME": "Anna Schmidt", "DES": "Head of Operations", "ORG": "Nordlicht Logistik GmbH",
     "PHONE": "+49 30 1234 5678", "EMAIL": "anna.schmidt@nordlicht.example.com", "WEB": "www.nordlicht.example.com",
     "style": {"bg": (250, 250, 247), "ink": (25, 35, 45), "accent": (20, 90, 160)}},
    {"NAME": "Lukas Becker", "DES": "Senior Data Engineer", "ORG": "Rheinwerk Analytics AG",
     "PHONE": "+49 221 987 6543", "EMAIL": "l.becker@rheinwerk.example.com", "WEB": "www.rheinwerk.example.com",
     "style": {"bg": (245, 247, 250), "ink": (30, 30, 30), "accent": (170, 50, 40)}},
    {"NAME": "Sophie Wagner", "DES": "Marketing Manager", "ORG": "Alpenblick Design Studio",
     "PHONE": "+49 89 5550 1234", "EMAIL": "sophie@alpenblick.example.com", "WEB": "www.alpenblick.example.com",
     "style": {"bg": (252, 249, 244), "ink": (40, 40, 40), "accent": (40, 120, 80)}},
]


def font(name: str, size: int):
    try:
        return ImageFont.truetype(str(FONT_DIR / name), size)
    except OSError:
        return ImageFont.load_default()


def render_card(card: dict, w: int = 1050, h: int = 600) -> np.ndarray:
    st = card["style"]
    img = Image.new("RGB", (w, h), st["bg"])
    d = ImageDraw.Draw(img)
    d.rectangle([0, 0, 22, h], fill=st["accent"])
    d.text((80, 70), card["ORG"].upper(), font=font("DejaVuSans-Bold.ttf", 34), fill=st["accent"])
    d.text((80, 210), card["NAME"], font=font("DejaVuSans-Bold.ttf", 58), fill=st["ink"])
    d.text((80, 285), card["DES"], font=font("DejaVuSans.ttf", 34), fill=st["ink"])
    small = font("DejaVuSans.ttf", 30)
    d.text((80, 400), f"Tel {card['PHONE']}", font=small, fill=st["ink"])
    d.text((80, 450), card["EMAIL"], font=small, fill=st["ink"])
    d.text((80, 500), card["WEB"], font=small, fill=st["ink"])
    return cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)


def place_on_desk(card: np.ndarray, seed: int, size=(1600, 1200), max_angle: float = 8) -> tuple[np.ndarray, np.ndarray]:
    """Put the card on a wooden-looking desk with a perspective tilt. Returns photo and true corners."""
    rng = np.random.default_rng(seed)
    W, H = size
    yy, xx = np.mgrid[0:H, 0:W]
    grain = (np.sin(xx / 9.0 + np.sin(yy / 60.0) * 3) * 10 + rng.normal(0, 6, (H, W))).astype(np.float32)
    desk = np.stack([70 + grain, 95 + grain, 125 + grain], axis=-1).clip(0, 255).astype(np.uint8)

    h, w = card.shape[:2]
    cx, cy = W / 2 + rng.uniform(-80, 80), H / 2 + rng.uniform(-60, 60)
    scale = rng.uniform(0.95, 1.1)
    hw, hh = w * scale / 2, h * scale / 2
    jitter = lambda: rng.uniform(-45, 45)  # noqa: E731
    dst = np.float32([[cx - hw + jitter(), cy - hh + jitter()], [cx + hw + jitter(), cy - hh + jitter()],
                      [cx + hw + jitter(), cy + hh + jitter()], [cx - hw + jitter(), cy + hh + jitter()]])
    angle = np.deg2rad(rng.uniform(-max_angle, max_angle))
    rot = np.array([[np.cos(angle), -np.sin(angle)], [np.sin(angle), np.cos(angle)]])
    dst = ((dst - [cx, cy]) @ rot.T + [cx, cy]).astype(np.float32)

    m = cv2.getPerspectiveTransform(np.float32([[0, 0], [w, 0], [w, h], [0, h]]), dst)
    shadow = cv2.warpPerspective(np.full((h, w), 255, np.uint8), m, (W, H))
    shadow = cv2.GaussianBlur(np.roll(shadow, (14, 10), axis=(0, 1)), (41, 41), 0)
    photo = (desk * (1 - 0.45 * shadow[..., None] / 255)).astype(np.uint8)
    warped = cv2.warpPerspective(card, m, (W, H))
    mask = cv2.warpPerspective(np.full((h, w), 255, np.uint8), m, (W, H))
    photo[mask > 0] = warped[mask > 0]
    light = np.linspace(1.05, 0.88, W, dtype=np.float32)[None, :, None]  # uneven lighting
    photo = (photo * light).clip(0, 255).astype(np.uint8)
    return photo, dst


INVOICE = {
    "seller": "Brandt & Söhne Bürobedarf GmbH",
    "invoice_number": "RE-2024-0187",
    "date": "14.03.2024",
    "net": "687,00",
    "vat_rate": "19",
    "vat": "130,53",
    "total": "817,53",
    "vat_id": "DE123456789",
    "iban": "DE89 3704 0044 0532 0130 00",
}


def render_invoice(w: int = 1240, h: int = 1754) -> np.ndarray:
    img = Image.new("RGB", (w, h), (252, 252, 250))
    d = ImageDraw.Draw(img)
    ink, grey, accent = (25, 28, 33), (95, 100, 110), (30, 70, 140)
    f = lambda size, bold=False: font("DejaVuSans-Bold.ttf" if bold else "DejaVuSans.ttf", size)  # noqa: E731
    d.text((100, 90), INVOICE["seller"], font=f(36, True), fill=accent)
    d.text((100, 140), "Hafenstraße 12 · 20457 Hamburg · Tel. 040 555 0187", font=f(22), fill=grey)
    d.text((100, 260), "Nordlicht Logistik GmbH", font=f(26), fill=ink)
    d.text((100, 295), "Musterweg 5", font=f(26), fill=ink)
    d.text((100, 330), "10115 Berlin", font=f(26), fill=ink)
    d.text((100, 460), "RECHNUNG", font=f(54, True), fill=ink)
    meta = [("Rechnungsnummer:", INVOICE["invoice_number"]), ("Rechnungsdatum:", INVOICE["date"]),
            ("Kundennummer:", "K-10452")]
    for i, (k, v) in enumerate(meta):
        d.text((720, 260 + i * 40), k, font=f(24), fill=grey)
        d.text((985, 260 + i * 40), v, font=f(24, True), fill=ink)

    y = 600
    cols = [(100, "Pos."), (190, "Beschreibung"), (700, "Menge"), (830, "Einzelpreis"), (1010, "Gesamt")]
    for x, t in cols:
        d.text((x, y), t, font=f(24, True), fill=ink)
    d.line([(100, y + 42), (1140, y + 42)], fill=ink, width=2)
    items = [("1", "Druckerpapier A4, Karton", "10", "24,90 €", "249,00 €"),
             ("2", "Toner XL schwarz", "4", "89,50 €", "358,00 €"),
             ("3", "Ordner breit, blau", "25", "3,20 €", "80,00 €")]
    for r, row in enumerate(items):
        yy = y + 70 + r * 50
        for (x, _), t in zip(cols, row):
            d.text((x, yy), t, font=f(24), fill=ink)
    y = y + 70 + len(items) * 50 + 20
    d.line([(700, y), (1140, y)], fill=grey, width=1)
    totals = [("Nettobetrag", f"{INVOICE['net']} €", False),
              (f"zzgl. {INVOICE['vat_rate']} % MwSt.", f"{INVOICE['vat']} €", False),
              ("Gesamtbetrag", f"{INVOICE['total']} €", True)]
    for i, (k, v, bold) in enumerate(totals):
        d.text((700, y + 25 + i * 48), k, font=f(26, bold), fill=ink)
        d.text((1010, y + 25 + i * 48), v, font=f(26, bold), fill=ink)
    d.text((100, 1350), "Zahlbar innerhalb von 14 Tagen ohne Abzug.", font=f(24), fill=ink)
    d.line([(100, 1560), (1140, 1560)], fill=grey, width=1)
    d.text((100, 1585), f"USt-IdNr.: {INVOICE['vat_id']}    Amtsgericht Hamburg HRB 99999", font=f(21), fill=grey)
    d.text((100, 1620), f"IBAN: {INVOICE['iban']}    BIC: COBADEFFXXX", font=f(21), fill=grey)
    return cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)


ARCHIVE_TEXT = [
    "STADTVERWALTUNG MUSTERSTADT",
    "Bauamt",
    "",
    "Musterstadt, den 12. Mai 1978",
    "",
    "Betreff: Baugenehmigung Nr. 78/1142",
    "",
    "Sehr geehrter Herr Hoffmann,",
    "",
    "auf Ihren Antrag vom 3. April 1978 wird Ihnen hiermit",
    "die Genehmigung zur Errichtung einer Garage auf dem",
    "Grundstück Lindenallee 14, Flurstück 233/7, erteilt.",
    "",
    "Die Bauarbeiten sind innerhalb von zwei Jahren nach",
    "Zugang dieses Bescheides zu beginnen. Die beigefügten",
    "Auflagen sind Bestandteil dieser Genehmigung.",
    "",
    "Hochachtungsvoll",
    "",
    "Im Auftrag",
    "K. Lehmann",
]


def render_archive(w: int = 1240, h: int = 1754, seed: int = 7) -> np.ndarray:
    """A typewritten 1970s letter on yellowed, stained paper with faded ink."""
    rng = np.random.default_rng(seed)
    img = Image.new("RGB", (w, h), (236, 224, 192))
    d = ImageDraw.Draw(img)
    mono = font("DejaVuSansMono.ttf", 30)
    for i, line in enumerate(ARCHIVE_TEXT):
        d.text((130 + rng.integers(-2, 3), 170 + i * 52), line, font=mono, fill=(58, 52, 48))
    arr = np.array(img).astype(np.float32)
    yy, xx = np.mgrid[0:h, 0:w]
    for _ in range(5):  # coffee and water stains
        cx, cy, r = rng.uniform(0, w), rng.uniform(0, h), rng.uniform(80, 260)
        ring = np.exp(-((np.hypot(xx - cx, yy - cy) - r) ** 2) / (2 * 12 ** 2))
        arr -= ring[..., None] * np.array([18, 26, 40]) * rng.uniform(0.4, 1.0)
    edge = np.minimum.reduce([xx, yy, w - xx, h - yy]).astype(np.float32)
    arr -= (np.exp(-edge / 60)[..., None] * np.array([25, 35, 55]))  # darker, older edges
    arr += rng.normal(0, 7, arr.shape)  # paper grain
    arr = cv2.GaussianBlur(arr.clip(0, 255).astype(np.uint8), (3, 3), 0)
    return cv2.cvtColor(arr, cv2.COLOR_RGB2BGR)


def main() -> None:
    out = ROOT / "samples"
    out.mkdir(exist_ok=True)
    for i, card in enumerate(CARDS, 1):
        photo, corners = place_on_desk(render_card(card), seed=i)
        cv2.imwrite(str(out / f"card_{i}.jpg"), photo, [cv2.IMWRITE_JPEG_QUALITY, 88])
        truth = {k: v for k, v in card.items() if k != "style"} | {"corners": corners.round(1).tolist()}
        (out / f"card_{i}.json").write_text(json.dumps(truth, indent=2))
        print("wrote", out / f"card_{i}.jpg")

    photo, corners = place_on_desk(render_invoice(), seed=11, size=(2000, 2600), max_angle=5)
    cv2.imwrite(str(out / "invoice_1.jpg"), photo, [cv2.IMWRITE_JPEG_QUALITY, 88])
    (out / "invoice_1.json").write_text(json.dumps(INVOICE | {"corners": corners.round(1).tolist()}, indent=2, ensure_ascii=False))
    print("wrote", out / "invoice_1.jpg")

    photo, corners = place_on_desk(render_archive(), seed=12, size=(2000, 2600), max_angle=5)
    cv2.imwrite(str(out / "archive_1.jpg"), photo, [cv2.IMWRITE_JPEG_QUALITY, 88])
    truth = {"lines": [ln for ln in ARCHIVE_TEXT if ln], "corners": corners.round(1).tolist()}
    (out / "archive_1.json").write_text(json.dumps(truth, indent=2, ensure_ascii=False))
    print("wrote", out / "archive_1.jpg")


if __name__ == "__main__":
    random.seed(0)
    main()
