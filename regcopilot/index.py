"""Hybrid retrieval: BM25 + dense embeddings, fused with Reciprocal Rank Fusion (RRF),
plus a direct lookup when the user names an Article ("What does DORA Article 28 say?")."""
from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from . import config
from .bm25 import BM25

RE_REF = re.compile(r"\b(?:(AI Act|DORA)\s+)?(Article|Art\.?|Annex)\s+(\d+[a-z]?|[IVXLC]+)\b(?:\s+(?:of\s+)?(?:the\s+)?(AI Act|DORA))?", re.I)


@dataclass
class Hit:
    chunk: dict
    score: float
    bm25_rank: int | None
    dense_rank: int | None


def load_chunks(path: Path | None = None) -> list[dict]:
    path = path or config.CHUNKS_PATH
    if not path.exists():
        raise FileNotFoundError(f"{path} not found. Run: python -m regcopilot ingest")
    return json.loads(path.read_text())


class Index:
    """`embed_backend` is "mistral", "ollama", "mock" or "none" (BM25 only)."""

    def __init__(self, chunks: list[dict], vectors: np.ndarray | None, embed_backend: str, embed_model: str = ""):
        self.chunks = chunks
        self.embed_backend = embed_backend
        self.embed_model = embed_model
        self.vectors = None
        if vectors is not None:
            norms = np.linalg.norm(vectors, axis=1, keepdims=True)
            self.vectors = vectors / np.maximum(norms, 1e-9)
        self.bm25 = BM25([c["text"] for c in chunks])
        self._client = None

    # ---------- build / persist ----------
    @classmethod
    def build(cls, chunks: list[dict], embed_backend: str, embed_model: str | None = None) -> "Index":
        vectors = None
        model = ""
        if embed_backend != "none":
            from .llm import get_client
            client = get_client(embed_backend)
            model = embed_model or client.default_embed_model
            vectors = client.embed([c["text"] for c in chunks], model=model)
        idx = cls(chunks, vectors, embed_backend, model)
        idx.save()
        return idx

    def save(self) -> Path:
        d = config.INDEX_DIR / self.embed_backend
        d.mkdir(parents=True, exist_ok=True)
        (d / "meta.json").write_text(json.dumps({
            "embed_backend": self.embed_backend, "embed_model": self.embed_model,
            "n_chunks": len(self.chunks), "chunk_ids": [c["chunk_id"] for c in self.chunks],
        }, indent=2))
        if self.vectors is not None:
            np.save(d / "vectors.npy", self.vectors)
        return d

    @classmethod
    def load(cls, embed_backend: str) -> "Index":
        chunks = load_chunks()
        if embed_backend == "none":
            return cls(chunks, None, "none")
        d = config.INDEX_DIR / embed_backend
        if not (d / "meta.json").exists():
            raise FileNotFoundError(f"No '{embed_backend}' index. Run: python -m regcopilot index --embed {embed_backend}")
        meta = json.loads((d / "meta.json").read_text())
        if meta["chunk_ids"] != [c["chunk_id"] for c in chunks]:
            raise RuntimeError("Index is out of date with data/chunks.json - rebuild it.")
        return cls(chunks, np.load(d / "vectors.npy"), embed_backend, meta["embed_model"])

    @staticmethod
    def available() -> list[str]:
        found = ["none"] if config.CHUNKS_PATH.exists() else []
        if config.INDEX_DIR.exists():
            found += sorted(p.name for p in config.INDEX_DIR.iterdir() if (p / "meta.json").exists())
        return found

    # ---------- search ----------
    def _query_vector(self, query: str) -> np.ndarray:
        if self._client is None:
            from .llm import get_client
            self._client = get_client(self.embed_backend)
        v = self._client.embed([query], model=self.embed_model)[0]
        return v / max(np.linalg.norm(v), 1e-9)

    def _explicit_refs(self, query: str) -> list[str]:
        """Unit ids the user named directly, e.g. 'DORA Article 28' -> DORA-Art-28."""
        q = query
        mentions_aia = bool(re.search(r"\bAI Act\b", q, re.I))
        mentions_dora = bool(re.search(r"\bDORA\b", q, re.I))
        ids = []
        for m in RE_REF.finditer(q):
            src_name = (m.group(1) or m.group(4) or "").lower()
            kind = "Annex" if m.group(2).lower() == "annex" else "Art"
            num = m.group(3).upper() if kind == "Annex" else m.group(3)
            if src_name:
                srcs = ["AIA"] if src_name == "ai act" else ["DORA"]
            elif mentions_aia != mentions_dora:
                srcs = ["AIA"] if mentions_aia else ["DORA"]
            else:
                srcs = ["AIA", "DORA"]
            ids += [f"{s}-{kind}-{num}" for s in srcs]
        return ids

    def search(self, query: str, k: int = config.TOP_K, pool: int = 50, rrf_k: int = 60) -> list[Hit]:
        bm = np.asarray(self.bm25.scores(query))
        bm_order = [i for i in np.argsort(-bm)[:pool] if bm[i] > 0]
        bm_rank = {i: r for r, i in enumerate(bm_order)}
        dn_rank: dict[int, int] = {}
        if self.vectors is not None:
            sims = self.vectors @ self._query_vector(query)
            dn_rank = {int(i): r for r, i in enumerate(np.argsort(-sims)[:pool])}

        fused: dict[int, float] = {}
        for ranks in (bm_rank, dn_rank):
            for i, r in ranks.items():
                fused[i] = fused.get(i, 0.0) + 1.0 / (rrf_k + r + 1)

        # Boost Articles/Annexes the user named explicitly.
        named = set(self._explicit_refs(query))
        if named:
            top = max(fused.values(), default=0.0) + 1.0
            for i, c in enumerate(self.chunks):
                if c["unit_id"] in named:
                    fused[i] = top + fused.get(i, 0.0)

        order = sorted(fused, key=lambda i: -fused[i])[:k]
        return [Hit(self.chunks[i], fused[i], bm_rank.get(i), dn_rank.get(i)) for i in order]
