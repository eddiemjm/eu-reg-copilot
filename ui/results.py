"""Read eval outputs from results/. Summaries are committed; per-question jsonl files may not be."""
from __future__ import annotations

import json
import re
from datetime import datetime
from pathlib import Path

RE_SUMMARY = re.compile(r"^(\d{8}-\d{6})_summary\.md$")
# Changes between runs that the charts should call out, keyed by the first run that includes them.
NOTES = {"20261011-185938": "Prompt: dates only from sources · parser reads (…) citations"}
TEXT_COLUMNS = {"model", "backend", "index", "tokens in/out"}


def _num(value: str):
    if value in ("", "n/a", "None"):
        return None
    try:
        return float(value)
    except ValueError:
        return value


def parse_summary(text: str) -> list[dict]:
    """Rows of a run_eval.py summary table, keyed by header name (old runs have fewer columns)."""
    lines = [l for l in text.splitlines() if l.startswith("|") and not l.startswith("|---")]
    if not lines:
        return []
    cells = lambda l: [c.strip() for c in l.strip().strip("|").split("|")]  # noqa: E731
    header = cells(lines[0])
    return [{h: (v if h in TEXT_COLUMNS else _num(v)) for h, v in zip(header, cells(l))} for l in lines[1:]]


def load_summaries(results_dir: Path) -> list[dict]:
    """Every summary row from every run, oldest first, each tagged with its run `stamp`."""
    rows = []
    for f in sorted(results_dir.glob("*_summary.md")):
        m = RE_SUMMARY.match(f.name)
        if m:
            rows += [{**r, "stamp": m.group(1)} for r in parse_summary(f.read_text())]
    return rows


def latest(rows: list[dict], model: str, index: str, before: str | None = None) -> dict | None:
    """Most recent row for this model and retrieval index (optionally from a run before `before`)."""
    match = [r for r in rows if r.get("model") == model and r.get("index") == index
             and (before is None or r["stamp"] < before)]
    return match[-1] if match else None


def load_rows(results_dir: Path, stamp: str, backend: str, model: str) -> list[dict] | None:
    f = results_dir / f"{stamp}_{backend}_{model.replace('/', '-')}.jsonl"
    if not f.exists():
        return None
    return [json.loads(l) for l in f.read_text().splitlines() if l.strip()]


def run_label(stamp: str) -> str:
    dt = datetime.strptime(stamp, "%Y%m%d-%H%M%S")
    return f"{dt.day} {dt:%b %H:%M}"
