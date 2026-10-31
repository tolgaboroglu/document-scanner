"""Evaluate the shipped extractor (NER + rules) on the held-out test cards.

Uses the same card-level split as train_ner.py and the same entity-level
exact-match metric, and adds the result to docs/ner_metrics.json.

    python scripts/evaluate.py legacy/business_card_/BusinessCardNER/businessCard.csv
"""

from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

from docscan.extract import merge_spans  # noqa: E402
from docscan.ner import CardNER  # noqa: E402
from docscan.rules import find_spans  # noqa: E402
from train_ner import load_cards, score  # noqa: E402


def spans_to_tags(n: int, spans) -> list[str]:
    tags = ["O"] * n
    for s, e, lab, *_ in spans:
        for i in range(s, e):
            tags[i] = f"{'B' if i == s else 'I'}-{lab}"
    return tags


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("csv", type=Path)
    ap.add_argument("--test-size", type=float, default=0.15)
    args = ap.parse_args()

    cards = load_cards(args.csv)
    ids = sorted(cards)
    random.Random(42).shuffle(ids)
    test_ids = ids[: max(1, int(len(ids) * args.test_size))]

    ner = CardNER(ROOT / "models" / "card_ner")
    gold, rules_only, combined = [], [], []
    for cid in test_ids:
        words = [w for w, _ in cards[cid]]
        gold.append([t for _, t in cards[cid]])
        rule_spans = find_spans(words)
        rules_only.append(spans_to_tags(len(words), rule_spans))
        combined.append(spans_to_tags(len(words), merge_spans(ner.predict(words), rule_spans)))

    path = ROOT / "docs" / "ner_metrics.json"
    metrics = json.loads(path.read_text()) if path.exists() else {}
    metrics["rules_only"] = score(gold, rules_only)
    metrics["v2_ner_plus_rules"] = score(gold, combined)
    path.write_text(json.dumps(metrics, indent=2))
    for key in ("v1_label_scheme", "v2_fixed_labels", "rules_only", "v2_ner_plus_rules"):
        if key in metrics:
            print(f"{key:20s}", {k: v["f1"] for k, v in metrics[key].items()})


if __name__ == "__main__":
    main()
