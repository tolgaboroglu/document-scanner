"""Tests run on synthetic sample cards; no real personal data and no API keys needed."""

import json
import shutil
from pathlib import Path

import cv2
import numpy as np
import pytest

from docscan import scanner
from docscan.export import to_vcard
from docscan.extract import Contact, Field, merge_spans
from docscan.llm import ClaudeExtractor, parse_llm_json
from docscan.ner import bio_to_entities
from docscan.rules import find_spans, normalise_phone

ROOT = Path(__file__).resolve().parent.parent
SAMPLES = sorted((ROOT / "samples").glob("card_*.jpg"))
needs_tesseract = pytest.mark.skipif(shutil.which("tesseract") is None, reason="Tesseract not installed")


def truth(path: Path) -> dict:
    return json.loads(path.with_suffix(".json").read_text())


# --- pure logic -------------------------------------------------------------

def test_bio_to_entities():
    tags = ["B-NAME", "I-NAME", "O", "B-ORG", "I-ORG", "I-PHONE", "B-PHONE", "B-PHONE"]
    assert bio_to_entities(tags) == [(0, 2, "NAME"), (3, 5, "ORG"), (5, 6, "PHONE"), (6, 7, "PHONE"), (7, 8, "PHONE")]


def test_rules_find_structured_fields():
    words = ["Tel", "+49", "30", "1234", "5678", "a.b@firma.example.com", "www.firma.example.com", "Berlin"]
    assert find_spans(words) == [(1, 5, "PHONE"), (5, 6, "EMAIL"), (6, 7, "WEB")]


def test_phone_comma_ends_number():
    assert find_spans(["040-4852", "8881,", "90309", "52549"]) == [(0, 2, "PHONE"), (2, 4, "PHONE")]


def test_normalise_phone_keeps_plus():
    assert normalise_phone("+49 (30) 1234-5678") == "+493012345678"


def test_rules_only_fill_gaps():
    ner = [(0, 2, "NAME"), (5, 6, "WEB")]
    rules = [(5, 6, "EMAIL"), (7, 9, "PHONE")]
    assert merge_spans(ner, rules) == [(0, 2, "NAME", "ner"), (5, 6, "WEB", "ner"), (7, 9, "PHONE", "rules")]


def test_order_points():
    pts = np.array([[10, 90], [90, 10], [10, 10], [90, 90]], dtype=np.float32)
    assert scanner.order_points(pts).tolist() == [[10, 10], [90, 10], [90, 90], [10, 90]]


def test_vcard():
    contact = Contact([Field("NAME", "Anna Schmidt"), Field("ORG", "Nordlicht, GmbH"),
                       Field("PHONE", "+493012345678"), Field("WEB", "www.example.com")])
    card = to_vcard(contact)
    assert "N:Schmidt;Anna;;;" in card and "ORG:Nordlicht\\, GmbH" in card
    assert "TEL;TYPE=WORK,VOICE:+493012345678" in card and "URL:https://www.example.com" in card
    assert card.startswith("BEGIN:VCARD\r\n") and card.endswith("END:VCARD\r\n")


def test_llm_json_is_validated():
    reply = 'Sure:\n{"NAME": ["Anna Schmidt"], "EMAIL": ["not-an-email", "anna@x.example.com"], "PHONE": ["12"]}'
    contact = parse_llm_json(reply)
    assert contact.values("NAME") == ["Anna Schmidt"]
    assert contact.values("EMAIL") == ["anna@x.example.com"]
    assert contact.values("PHONE") == []


def test_llm_extractor_with_fake_client():
    class Block:
        type, text = "text", '{"NAME": ["Lukas Becker"], "ORG": ["Rheinwerk Analytics AG"]}'

    class Client:
        class messages:  # noqa: N801
            @staticmethod
            def create(**kwargs):
                assert kwargs["messages"][0]["content"][0]["type"] == "image"
                return type("R", (), {"content": [Block()]})

    contact = ClaudeExtractor(model="test-model", client=Client()).extract(np.zeros((60, 100, 3), np.uint8))
    assert contact.values("ORG") == ["Rheinwerk Analytics AG"]


# --- computer vision on the synthetic photos -----------------------------------

@pytest.mark.parametrize("path", SAMPLES, ids=lambda p: p.stem)
def test_detects_card_corners(path):
    image = cv2.imread(str(path))
    found = scanner.detect_corners(image)
    assert found is not None
    expected = scanner.order_points(np.array(truth(path)["corners"]))
    assert np.abs(found - expected).max() < 30  # pixels, on a 1600×1200 photo


def test_no_card_returns_none():
    noise = np.random.default_rng(0).integers(0, 255, (600, 800, 3), dtype=np.uint8)
    assert scanner.detect_corners(cv2.GaussianBlur(noise, (21, 21), 0)) is None


# --- end to end ---------------------------------------------------------------

@pytest.fixture(scope="module")
def card_scanner():
    from docscan.pipeline import CardScanner

    return CardScanner()


@needs_tesseract
@pytest.mark.parametrize("path", SAMPLES, ids=lambda p: p.stem)
def test_end_to_end(card_scanner, path):
    t = truth(path)
    result = card_scanner.scan(cv2.imread(str(path)))
    c = result.contact
    assert normalise_phone(t["PHONE"]) in c.values("PHONE")
    assert t["WEB"] in c.values("WEB")
    assert t["NAME"] in c.values("NAME")


@needs_tesseract
def test_api_flow(card_scanner):
    from docscan.server import create_app

    client = create_app(card_scanner).test_client()
    assert client.get("/").status_code == 200
    det = client.post("/api/detect", data={"sample": SAMPLES[0].stem}).json
    assert det["found"] and len(det["corners"]) == 4
    ext = client.post("/api/extract", json={"id": det["id"], "corners": det["corners"]})
    assert ext.status_code == 200 and ext.json["fields"]
    vcf = client.post("/api/export/vcf", json={"fields": ext.json["fields"]})
    assert vcf.status_code == 200 and b"BEGIN:VCARD" in vcf.data
    assert client.post("/api/detect", data={}).status_code == 400
    assert client.post("/api/extract", json={"id": "missing", "corners": det["corners"]}).status_code == 404
