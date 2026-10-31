"""Optional: extract the same fields with a vision LLM (Claude).

Useful for unusual layouts or languages the NER model never saw. The model
receives the flattened card image and must answer with JSON only; values it
returns are validated with the same rules as the OCR path.
"""

from __future__ import annotations

import base64
import json
import os
import re

import cv2
import numpy as np

from .extract import Contact, Field, clean_value
from .ner import LABELS
from .rules import EMAIL

PROMPT = """Extract the contact details printed on this business card.
Reply with JSON only, using exactly these keys, each a list of strings:
{"NAME": [], "ORG": [], "DES": [], "PHONE": [], "EMAIL": [], "WEB": []}
DES is the job title. Copy text exactly as printed; do not guess missing values."""


class ClaudeExtractor:
    def __init__(self, model: str | None = None, client=None):
        if client is None:
            import anthropic

            if not os.getenv("ANTHROPIC_API_KEY"):
                raise RuntimeError("ANTHROPIC_API_KEY is not set")
            client = anthropic.Anthropic()
        self.client = client
        self.model = model or os.getenv("ANTHROPIC_MODEL", "claude-sonnet-5-5")
        self.name = f"Claude vision ({self.model})"

    def extract(self, image: np.ndarray) -> Contact:
        ok, buf = cv2.imencode(".jpg", image, [cv2.IMWRITE_JPEG_QUALITY, 90])
        if not ok:
            raise ValueError("could not encode image")
        resp = self.client.messages.create(
            model=self.model,
            max_tokens=600,
            messages=[{
                "role": "user",
                "content": [
                    {"type": "image", "source": {"type": "base64", "media_type": "image/jpeg",
                                                 "data": base64.b64encode(buf.tobytes()).decode()}},
                    {"type": "text", "text": PROMPT},
                ],
            }],
        )
        text = "".join(b.text for b in resp.content if getattr(b, "type", "") == "text")
        return parse_llm_json(text)


def parse_llm_json(text: str) -> Contact:
    match = re.search(r"\{.*\}", text, re.DOTALL)
    data = json.loads(match.group(0)) if match else {}
    contact = Contact()
    for label in LABELS:
        for raw in data.get(label) or []:
            value = clean_value(label, str(raw))
            if label == "EMAIL" and not EMAIL.match(value):
                continue
            if label == "PHONE" and not 7 <= len(value.lstrip("+")) <= 15:
                continue
            if value:
                contact.fields.append(Field(label, value, None, "llm"))
    return contact
