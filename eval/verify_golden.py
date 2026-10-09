"""Sanity-check the golden set against the ingested text: every expected Article/Annex must
exist, and must contain its verification phrase. Catches typos in the eval set and parser
regressions before they silently distort the metrics.

    python eval/verify_golden.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from regcopilot.index import load_chunks  # noqa: E402

GOLDEN = Path(__file__).with_name("golden.jsonl")


def load_golden(path: Path = GOLDEN) -> list[dict]:
    return [json.loads(l) for l in path.read_text().splitlines() if l.strip()]


def verify(golden: list[dict], chunks: list[dict]) -> list[str]:
    text_by_unit: dict[str, str] = {}
    for c in chunks:
        text_by_unit[c["unit_id"]] = text_by_unit.get(c["unit_id"], "") + "\n" + c["text"]
    problems = []
    for g in golden:
        for uid in g.get("expected", []):
            if uid not in text_by_unit:
                problems.append(f"{g['id']}: expected unit {uid} not found in corpus")
        for uid, phrase in g.get("verify", {}).items():
            body = text_by_unit.get(uid, "")
            if body and phrase.lower() not in body.lower():
                problems.append(f"{g['id']}: '{phrase}' not found in {uid}")
    return problems


if __name__ == "__main__":
    golden = load_golden()
    problems = verify(golden, load_chunks())
    for p in problems:
        print("FAIL", p)
    print(f"{len(golden)} questions checked, {len(problems)} problem(s)")
    sys.exit(1 if problems else 0)
