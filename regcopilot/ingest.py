"""Fetch the regulations (see fetch.py) and turn them into chunks."""
from __future__ import annotations

import json
from collections import Counter
from pathlib import Path

from . import config
from .fetch import fetch
from .parse import chunk_units, parse_units
from .sources import SOURCES


def ingest(sources: list[str] | None = None, html_paths: dict[str, Path] | None = None,
           include_recitals: bool = False, max_words: int = config.MAX_CHUNK_WORDS,
           out: Path | None = None, refresh: bool = False, verbose: bool = False) -> list[dict]:
    out = out or config.CHUNKS_PATH
    sources = sources or list(SOURCES)
    all_chunks: list[dict] = []
    for src in sources:
        path = (html_paths or {}).get(src) or fetch(src, refresh=refresh, verbose=verbose)
        units = parse_units(Path(path).read_text(encoding="utf-8"), src, include_recitals=include_recitals)
        chunks = chunk_units(units, max_words=max_words)
        kinds = Counter(u.kind for u in units)
        print(f"{SOURCES[src]['short']:7} {kinds.get('article', 0):4} articles  {kinds.get('annex', 0):3} annexes  "
              f"{kinds.get('recital', 0):4} recitals  -> {len(chunks)} chunks")
        all_chunks += [c.to_dict() for c in chunks]
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(all_chunks, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"Wrote {len(all_chunks)} chunks to {out}")
    return all_chunks
