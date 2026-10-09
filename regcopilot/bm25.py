"""Small, dependency-free BM25 (Okapi) implementation.

Keyword search matters for legal text: users type exact terms ("register of information",
"threat-led penetration testing") that dense embeddings can blur. We fuse it with dense
retrieval in index.py.
"""
from __future__ import annotations

import math
import re
from collections import Counter

STOPWORDS = set(
    "a an and are as at be by can do does for from has have how i if in into is it its may "
    "must of on or shall should such that the their there these this those to under was what "
    "when where which who whom will with within would you your".split()
)
TOKEN = re.compile(r"[a-z0-9]+(?:[-'][a-z0-9]+)*")


def _stem(t: str) -> str:
    """Very light suffix stripping so 'penalties'~'penalty', 'emotions'~'emotion', 'testing'~'test'."""
    if len(t) <= 4 or t[0].isdigit():
        return t
    for suf, rep in (("ies", "y"), ("ing", ""), ("ed", ""), ("es", "e"), ("s", "")):
        if t.endswith(suf) and len(t) - len(suf) >= 3 and not t.endswith("ss"):
            return t[: -len(suf)] + rep
    return t


def tokenize(text: str) -> list[str]:
    return [_stem(t) for t in TOKEN.findall(text.lower()) if t not in STOPWORDS]


class BM25:
    def __init__(self, docs: list[str], k1: float = 1.5, b: float = 0.75):
        self.k1, self.b = k1, b
        self.docs = [tokenize(d) for d in docs]
        self.tf = [Counter(d) for d in self.docs]
        self.len = [len(d) for d in self.docs]
        self.avgdl = sum(self.len) / max(len(self.len), 1)
        df: Counter = Counter()
        for d in self.tf:
            df.update(d.keys())
        n = len(self.docs)
        self.idf = {t: math.log(1 + (n - f + 0.5) / (f + 0.5)) for t, f in df.items()}

    def scores(self, query: str) -> list[float]:
        q = tokenize(query)
        out = []
        for tf, dl in zip(self.tf, self.len):
            s = 0.0
            for t in q:
                if t in tf:
                    f = tf[t]
                    s += self.idf[t] * f * (self.k1 + 1) / (f + self.k1 * (1 - self.b + self.b * dl / self.avgdl))
            out.append(s)
        return out
