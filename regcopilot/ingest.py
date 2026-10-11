"""Fetch the regulations (see fetch.py) and turn them into chunks.

If the live fetch fails for a source, ingest falls back to the snapshot of the corpus
committed with the repository (corpus/, with its provenance manifest) rather than failing
the whole run - the manifest records it as 'committed-snapshot'."""
from __future__ import annotations

import json
from collections import Counter
from pathlib import Path

from . import config
from .fetch import fetch
from .parse import chunk_units, parse_units
from .sources import SOURCES


def _snapshot(source: str) -> tuple[list[dict], dict] | None:
    """Chunks for `source` from the committed corpus snapshot, plus its manifest entry."""
    try:
        chunks = json.loads(config.SNAPSHOT_CHUNKS.read_text())
        manifest = json.loads(config.SNAPSHOT_MANIFEST.read_text())
    except (OSError, json.JSONDecodeError):
        return None
    mine = [c for c in chunks if c.get("source") == source]
    if not mine or source not in manifest:
        return None
    return mine, manifest[source]


def ingest(sources: list[str] | None = None, html_paths: dict[str, Path] | None = None,
           include_recitals: bool = False, max_words: int = config.MAX_CHUNK_WORDS,
           out: Path | None = None, refresh: bool = False, verbose: bool = False) -> list[dict]:
    out = out or config.CHUNKS_PATH
    sources = sources or list(SOURCES)
    all_chunks: list[dict] = []
    fell_back: list[str] = []
    for src in sources:
        supplied = (html_paths or {}).get(src)
        chunks: list[dict]
        try:
            path = supplied or fetch(src, refresh=refresh, verbose=verbose)
            units = parse_units(Path(path).read_text(encoding="utf-8"), src, include_recitals=include_recitals)
            kinds = Counter(u.kind for u in units)
            chunks = [c.to_dict() for c in chunk_units(units, max_words=max_words)]
        except Exception as e:
            # The snapshot has no recitals, and a caller-supplied file failing is their problem to see.
            if supplied is not None or include_recitals:
                raise
            snap = _snapshot(src)
            if snap is None:
                raise
            chunks, entry = snap
            fell_back.append(src)
            print(f"{SOURCES[src]['short']:7} fetch failed ({type(e).__name__}); using the committed snapshot "
                  f"(text fetched {entry['fetched_at'][:10]} via {entry['strategy']}, "
                  f"SHA-256 {entry['sha256'][:12]}…)  -> {len(chunks)} chunks")
        else:
            print(f"{SOURCES[src]['short']:7} {kinds.get('article', 0):4} articles  {kinds.get('annex', 0):3} annexes  "
                  f"{kinds.get('recital', 0):4} recitals  -> {len(chunks)} chunks")
        all_chunks += chunks
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(all_chunks, ensure_ascii=False, indent=1), encoding="utf-8")
    if fell_back:
        config.RAW_DIR.mkdir(parents=True, exist_ok=True)
        mpath = config.RAW_DIR / "manifest.json"
        manifest = json.loads(mpath.read_text()) if mpath.exists() else {}
        snap_manifest = json.loads(config.SNAPSHOT_MANIFEST.read_text())
        for src in fell_back:
            entry = dict(snap_manifest[src])
            entry["strategy"] = "committed-snapshot"
            manifest[src] = entry
        mpath.write_text(json.dumps(manifest, indent=2))
    print(f"Wrote {len(all_chunks)} chunks to {out}")
    return all_chunks
