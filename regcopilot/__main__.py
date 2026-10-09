"""Command line:

    python -m regcopilot fetch -v               # just download (and show which route worked)
    python -m regcopilot ingest                 # fetch if needed, then parse + chunk the AI Act and DORA
    python -m regcopilot index --embed mistral  # build embeddings (mistral | ollama | none)
    python -m regcopilot ask "Which insurers need a fundamental rights impact assessment?" --backend mistral
"""
from __future__ import annotations

import argparse
import sys
import textwrap
from pathlib import Path

from . import config


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(prog="regcopilot", description="EU AI Act & DORA copilot")
    sub = p.add_subparsers(dest="cmd", required=True)

    pf = sub.add_parser("fetch", help="download the official texts (Cellar / EUR-Lex)")
    pf.add_argument("--refresh", action="store_true", help="re-download even if a valid copy is cached")
    pf.add_argument("-v", "--verbose", action="store_true", help="show every attempt")

    pi = sub.add_parser("ingest", help="fetch (if needed) and chunk the regulations")
    pi.add_argument("--refresh", action="store_true", help="re-download the texts first")
    pi.add_argument("-v", "--verbose", action="store_true")
    pi.add_argument("--html", nargs=2, action="append", metavar=("SOURCE", "PATH"),
                    help="use a local HTML file, e.g. --html AIA ./ai_act.html")
    pi.add_argument("--recitals", action="store_true", help="also index recitals")
    pi.add_argument("--max-words", type=int, default=config.MAX_CHUNK_WORDS)

    px = sub.add_parser("index", help="build the embedding index")
    px.add_argument("--embed", default="mistral", choices=["mistral", "ollama", "mock", "none"])
    px.add_argument("--model", default=None, help="embedding model override")

    pa = sub.add_parser("ask", help="ask a question")
    pa.add_argument("question")
    pa.add_argument("--backend", default="mistral", choices=["mistral", "ollama", "mock"])
    pa.add_argument("--model", default=None)
    pa.add_argument("--embed", default=None, help="index to use (defaults to the backend's, else BM25)")
    pa.add_argument("-k", type=int, default=config.TOP_K)

    args = p.parse_args(argv)

    if args.cmd == "fetch":
        from .fetch import fetch
        from .sources import SOURCES
        for src in SOURCES:
            path = fetch(src, refresh=args.refresh, verbose=args.verbose)
            print(f"{SOURCES[src]['short']}: ready at {path}")
    elif args.cmd == "ingest":
        from .ingest import ingest
        html = {s.upper(): Path(path) for s, path in (args.html or [])}
        ingest(html_paths=html, include_recitals=args.recitals, max_words=args.max_words,
               refresh=args.refresh, verbose=args.verbose)
    elif args.cmd == "index":
        from .index import Index, load_chunks
        idx = Index.build(load_chunks(), args.embed, args.model)
        print(f"Built '{args.embed}' index over {len(idx.chunks)} chunks ({idx.embed_model or 'BM25 only'})")
    elif args.cmd == "ask":
        from .index import Index
        from .llm import get_client
        from .rag import answer
        embed = args.embed or (args.backend if args.backend in Index.available() else "none")
        idx = Index.load(embed)
        a = answer(args.question, idx, get_client(args.backend), model=args.model, k=args.k)
        print("\n" + textwrap.fill(a.text, 100, replace_whitespace=False) + "\n")
        print("Retrieved:", ", ".join(a.retrieved_units))
        if a.ungrounded_citations:
            print("WARNING - cited but not retrieved:", ", ".join(a.ungrounded_citations))
        print(f"{a.model} | {a.prompt_tokens}+{a.completion_tokens} tokens | {a.latency_s:.1f}s | index: {embed}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
