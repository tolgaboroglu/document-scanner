"""Flask web app: choose a document type → upload → adjust corners → review → download.

Run:  python -m docscan.server        (http://127.0.0.1:5000)
"""

from __future__ import annotations

import base64
import logging
import os
import uuid
from collections import OrderedDict
from pathlib import Path

import cv2
import numpy as np
from flask import Flask, Response, jsonify, render_template, request

from . import scanner
from .export import to_csv, to_json, to_vcard
from .extract import Contact, Field
from .ner import LABELS
from .pipeline import DocScanner

ROOT = Path(__file__).resolve().parent.parent
SAMPLES = ROOT / "samples"
MAX_SIDE = 2400
MAX_BYTES = 12 * 1024 * 1024
log = logging.getLogger("docscan")


class ImageStore:
    """Keeps the last few uploads in memory so corner edits do not need a re-upload."""

    def __init__(self, size: int = 32):
        self.size = size
        self.items: OrderedDict[str, np.ndarray] = OrderedDict()

    def put(self, image: np.ndarray) -> str:
        key = uuid.uuid4().hex
        self.items[key] = image
        while len(self.items) > self.size:
            self.items.popitem(last=False)
        return key

    def get(self, key: str) -> np.ndarray | None:
        return self.items.get(key)


def decode_image(data: bytes) -> np.ndarray | None:
    image = cv2.imdecode(np.frombuffer(data, np.uint8), cv2.IMREAD_COLOR)
    if image is None:
        return None
    h, w = image.shape[:2]
    if max(h, w) > MAX_SIDE:  # keep phone photos fast to process
        f = MAX_SIDE / max(h, w)
        image = cv2.resize(image, (int(w * f), int(h * f)), interpolation=cv2.INTER_AREA)
    return image


def data_url(image: np.ndarray, quality: int = 85) -> str:
    ok, buf = cv2.imencode(".jpg", image, [cv2.IMWRITE_JPEG_QUALITY, quality])
    return "data:image/jpeg;base64," + base64.b64encode(buf.tobytes()).decode()


def create_app(card_scanner: DocScanner | None = None) -> Flask:
    app = Flask(__name__, template_folder=str(ROOT / "templates"), static_folder=str(ROOT / "static"))
    app.config["MAX_CONTENT_LENGTH"] = MAX_BYTES
    engine = card_scanner or DocScanner()
    store = ImageStore()
    pdfs: OrderedDict[str, bytes] = OrderedDict()

    @app.get("/")
    def index():
        samples = {kind: sorted(p.stem for p in SAMPLES.glob(f"{kind}_*.jpg")) for kind in ("card", "invoice", "archive")}
        return render_template("index.html", samples=samples, engine=engine.name)

    @app.get("/samples/<name>.jpg")
    def sample_image(name: str):
        path = SAMPLES / f"{Path(name).name}.jpg"
        if not path.is_file():
            return jsonify(error="Sample not found."), 404
        return Response(path.read_bytes(), mimetype="image/jpeg")

    @app.post("/api/detect")
    def detect():
        sample = request.form.get("sample")
        if sample:
            path = SAMPLES / f"{Path(sample).name}.jpg"
            data = path.read_bytes() if path.is_file() else b""
        else:
            upload = request.files.get("image")
            data = upload.read() if upload else b""
        image = decode_image(data) if data else None
        if image is None:
            return jsonify(error="Upload a JPG or PNG photo of a business card."), 400
        corners = scanner.detect_corners(image)
        found = corners is not None
        if not found:
            corners = scanner.default_corners(image)
        h, w = image.shape[:2]
        return jsonify(id=store.put(image), width=w, height=h, found=found,
                       corners=corners.round(1).tolist(), image=data_url(image, 80))

    @app.post("/api/extract")
    def extract():
        body = request.get_json(silent=True) or {}
        image = store.get(str(body.get("id", "")))
        if image is None:
            return jsonify(error="This photo has expired. Upload it again."), 404
        try:
            corners = np.array(body["corners"], dtype=np.float32).reshape(4, 2)
        except (KeyError, ValueError, TypeError):
            return jsonify(error="Corners must be four [x, y] points."), 400
        mode = body.get("mode", "card")
        try:
            if mode == "invoice":
                res = engine.invoice(image, corners)
                return jsonify(mode=mode, card=data_url(res.page), text=res.text, invoice=res.invoice.to_dict(),
                               engine="Tesseract 5 + invoice rules")
            if mode == "archive":
                res = engine.archive(image, corners)
                key = uuid.uuid4().hex
                pdfs[key] = res.pdf
                while len(pdfs) > 16:
                    pdfs.popitem(last=False)
                return jsonify(mode=mode, card=data_url(res.page), text=res.text, pdf=f"/api/pdf/{key}",
                               engine="Background removal + Tesseract 5")
            result = engine.scan(image, corners)
        except Exception:
            log.exception("extract failed")
            return jsonify(error="Text extraction failed. Check that Tesseract is installed."), 500
        return jsonify(
            mode="card",
            card=data_url(result.annotated),
            fields=result.contact.to_dict()["fields"],
            words=len(result.words),
            engine=engine.name,
        )

    @app.get("/api/pdf/<key>")
    def pdf(key: str):
        data = pdfs.get(key)
        if data is None:
            return jsonify(error="This PDF has expired. Scan the page again."), 404
        return Response(data, mimetype="application/pdf",
                        headers={"Content-Disposition": "attachment; filename=scan.pdf"})

    @app.post("/api/export/<fmt>")
    def export(fmt: str):
        body = request.get_json(silent=True) or {}
        contact = Contact([
            Field(str(f.get("label")), str(f.get("value", "")).strip(), None, str(f.get("source", "user")))
            for f in body.get("fields", [])
            if f.get("label") in LABELS and str(f.get("value", "")).strip()
        ])
        if fmt == "vcf":
            return Response(to_vcard(contact), mimetype="text/vcard",
                            headers={"Content-Disposition": "attachment; filename=contact.vcf"})
        if fmt == "json":
            return Response(to_json(contact), mimetype="application/json",
                            headers={"Content-Disposition": "attachment; filename=contact.json"})
        if fmt == "csv":
            return Response(to_csv(contact), mimetype="text/csv",
                            headers={"Content-Disposition": "attachment; filename=contact.csv"})
        return jsonify(error="Format must be vcf, json or csv."), 400

    @app.errorhandler(413)
    def too_large(_):
        return jsonify(error="The photo is larger than 12 MB."), 413

    return app


def main() -> None:
    try:
        from dotenv import load_dotenv

        load_dotenv()
    except ImportError:
        pass
    logging.basicConfig(level=logging.INFO)
    create_app().run(host=os.getenv("HOST", "127.0.0.1"), port=int(os.getenv("PORT", "5000")))


if __name__ == "__main__":
    main()
