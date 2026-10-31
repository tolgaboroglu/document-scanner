"""Train and evaluate the business-card NER model.

The v1 model was trained with every BIO tag used as its own entity label
("B-PHONE", "I-PHONE"), one entity per word, and with a few typos in the tags
("B-PHOE", "I-OR"). This script fixes the data, trains on the same split with
both label schemes, and evaluates both with the same entity-level metric:

    python scripts/train_ner.py legacy/business_card_/BusinessCardNER/businessCard.csv

Outputs:
    models/card_ner/          the fixed model used by the app
    docs/ner_metrics.json     per-label precision / recall / F1 for both schemes
"""

from __future__ import annotations

import argparse
import csv
import json
import random
import shutil
import sys
import tempfile
from collections import defaultdict
from pathlib import Path

import spacy
from spacy.cli.init_config import init_config
from spacy.cli.train import train as spacy_train
from spacy.tokens import Doc, DocBin, Span

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from docscan.ner import LABELS, bio_to_entities  # noqa: E402

TAG_FIXES = {"B-PHOE": "B-PHONE", "I-OR": "I-ORG", "I-DE": "I-DES"}


def normalise_tag(tag: str) -> str:
    tag = TAG_FIXES.get(tag.strip(), tag.strip())
    if tag in LABELS:  # bare "ORG" etc. → a new entity
        return f"B-{tag}"
    return tag if tag == "O" or tag[2:] in LABELS else "O"


def load_cards(path: Path) -> dict[str, list[tuple[str, str]]]:
    cards: dict[str, list[tuple[str, str]]] = defaultdict(list)
    with open(path, encoding="utf-8", errors="ignore", newline="") as fh:
        for row in csv.DictReader(fh):
            word = (row.get("text") or "").strip()
            if not word or not any(ch.isalnum() for ch in word):
                continue
            cards[row["id"]].append((word, normalise_tag(row.get("tag") or "O")))
    return {k: v for k, v in cards.items() if v}


def make_doc(nlp, tokens: list[tuple[str, str]], legacy: bool) -> Doc:
    words = [w for w, _ in tokens]
    tags = [t for _, t in tokens]
    doc = Doc(nlp.vocab, words=words)
    if legacy:  # v1 scheme: one entity per word, labelled with the full BIO tag
        doc.ents = [Span(doc, i, i + 1, label=t) for i, t in enumerate(tags) if t != "O"]
    else:
        doc.ents = [Span(doc, s, e, label=lab) for s, e, lab in bio_to_entities(tags)]
    return doc


def predicted_tags(doc: Doc, legacy: bool) -> list[str]:
    tags = ["O"] * len(doc)
    for ent in doc.ents:
        if legacy:  # "B-PHONE" → B / PHONE for the word it covers
            prefix, _, label = ent.label_.partition("-")
            for i in range(ent.start, ent.end):
                tags[i] = f"{prefix if i == ent.start else 'I'}-{label}"
        else:
            for i in range(ent.start, ent.end):
                tags[i] = f"{'B' if i == ent.start else 'I'}-{ent.label_}"
    return tags


def score(gold: list[list[str]], pred: list[list[str]]) -> dict:
    tp, fp, fn = defaultdict(int), defaultdict(int), defaultdict(int)
    for g, p in zip(gold, pred):
        gs, ps = set(bio_to_entities(g)), set(bio_to_entities(p))
        for _, _, lab in gs & ps:
            tp[lab] += 1
        for _, _, lab in ps - gs:
            fp[lab] += 1
        for _, _, lab in gs - ps:
            fn[lab] += 1

    def prf(t, f_p, f_n):
        p = t / (t + f_p) if t + f_p else 0.0
        r = t / (t + f_n) if t + f_n else 0.0
        return {"precision": round(p, 3), "recall": round(r, 3), "f1": round(2 * p * r / (p + r), 3) if p + r else 0.0}

    out = {lab: prf(tp[lab], fp[lab], fn[lab]) for lab in LABELS}
    out["overall"] = prf(sum(tp.values()), sum(fp.values()), sum(fn.values()))
    return out


def train_scheme(train_docs, dev_docs, output: Path, legacy: bool, max_steps: int) -> spacy.Language:
    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        DocBin(docs=train_docs).to_disk(tmp / "train.spacy")
        DocBin(docs=dev_docs).to_disk(tmp / "dev.spacy")
        config = init_config(lang="en", pipeline=["ner"], optimize="efficiency")
        config.to_disk(tmp / "config.cfg")
        overrides = {
            "paths.train": str(tmp / "train.spacy"),
            "paths.dev": str(tmp / "dev.spacy"),
            "training.max_steps": max_steps,
            "training.eval_frequency": 200,
            "training.seed": 0,
        }
        out = tmp / "out"
        spacy_train(tmp / "config.cfg", out, overrides=overrides)
        if output.exists():
            shutil.rmtree(output)
        shutil.copytree(out / "model-best", output)
    return spacy.load(output)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("csv", type=Path)
    ap.add_argument("--test-size", type=float, default=0.15)
    ap.add_argument("--max-steps", type=int, default=3000)
    args = ap.parse_args()

    cards = load_cards(args.csv)
    ids = sorted(cards)
    random.Random(42).shuffle(ids)
    n_test = max(1, int(len(ids) * args.test_size))
    test_ids, rest = ids[:n_test], ids[n_test:]
    dev_ids, train_ids = rest[:n_test], rest[n_test:]  # dev picks the best checkpoint; test is never seen
    print(f"{len(ids)} cards → {len(train_ids)} train / {len(dev_ids)} dev / {len(test_ids)} test (split by card)")

    blank = spacy.blank("en")
    gold = [[t for _, t in cards[i]] for i in test_ids]
    results = {"cards": {"train": len(train_ids), "dev": len(dev_ids), "test": len(test_ids)}}

    for name, legacy, out in (("v1_label_scheme", True, ROOT / "models" / "_legacy_scheme"),
                              ("v2_fixed_labels", False, ROOT / "models" / "card_ner")):
        train_docs = [make_doc(blank, cards[i], legacy) for i in train_ids]
        dev_docs = [make_doc(blank, cards[i], legacy) for i in dev_ids]
        nlp = train_scheme(train_docs, dev_docs, out, legacy, args.max_steps)
        preds = [predicted_tags(nlp(Doc(nlp.vocab, words=[w for w, _ in cards[i]])), legacy) for i in test_ids]
        results[name] = score(gold, preds)
        print(name, results[name]["overall"])
    shutil.rmtree(ROOT / "models" / "_legacy_scheme", ignore_errors=True)

    (ROOT / "docs").mkdir(exist_ok=True)
    (ROOT / "docs" / "ner_metrics.json").write_text(json.dumps(results, indent=2))
    print("Saved models/card_ner and docs/ner_metrics.json")


if __name__ == "__main__":
    main()
