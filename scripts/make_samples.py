"""Generate synthetic business-card photos for demos and tests.

All people, companies, numbers and domains are invented (domains use the
reserved example.com), so the repository ships no real personal data.

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


def place_on_desk(card: np.ndarray, seed: int, size=(1600, 1200)) -> tuple[np.ndarray, np.ndarray]:
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
    angle = np.deg2rad(rng.uniform(-8, 8))
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


def main() -> None:
    out = ROOT / "samples"
    out.mkdir(exist_ok=True)
    for i, card in enumerate(CARDS, 1):
        photo, corners = place_on_desk(render_card(card), seed=i)
        cv2.imwrite(str(out / f"card_{i}.jpg"), photo, [cv2.IMWRITE_JPEG_QUALITY, 88])
        truth = {k: v for k, v in card.items() if k != "style"} | {"corners": corners.round(1).tolist()}
        (out / f"card_{i}.json").write_text(json.dumps(truth, indent=2))
        print("wrote", out / f"card_{i}.jpg")


if __name__ == "__main__":
    random.seed(0)
    main()
