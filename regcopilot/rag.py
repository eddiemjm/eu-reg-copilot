"""Retrieve -> ground -> answer with citations -> verify citations."""
from __future__ import annotations

import re
from dataclasses import dataclass, field

from . import config
from .index import Hit, Index
from .sources import SHORT_TO_KEY, SOURCES

REFUSAL = "I can't find this in the AI Act or DORA texts I have."

SYSTEM_PROMPT = f"""You are a regulatory assistant for European financial-services teams.
You answer questions about two EU regulations: the AI Act (Regulation (EU) 2024/1689) and DORA (Regulation (EU) 2022/2554).

Rules:
1. Use ONLY the sources provided in the user message. Do not use outside knowledge.
2. Cite every claim with the source label in square brackets, exactly as written in the source header, optionally with a paragraph or point, e.g. [AI Act Art. 27(1)] or [DORA Art. 28(3)].
3. If the sources do not answer the question, reply with exactly: "{REFUSAL}" and nothing else.
4. Be precise about who an obligation applies to (provider, deployer, financial entity, ICT third-party service provider). State dates or deadlines only if they appear in the sources.
5. Be concise: at most 200 words. Use British English.
"""

# Matched anywhere in the answer, not only inside [...]: models also write "(DORA Art. 30(2))",
# and a citation the parser misses would also escape the grounding check.
RE_CITE = re.compile(r"\b(AI Act|DORA)\s+(?:(Articles?|Art\.?)\s*(\d+[a-z]?)|(Annex)\s+([IVXLC]+))\b", re.I)
# A follow-on citation that inherits the regulation: "AI Act Art. 16(a), Art. 26(5)", "... and Art. 28".
RE_CITE_CONT = re.compile(r"\s*(?:,|;|\band\b|&)\s*(?:(Articles?|Art\.?)\s*(\d+[a-z]?)|(Annex)\s+([IVXLC]+))\b", re.I)
# After a plural head, bare numbers continue the list: "DORA Articles 28 and 30".
RE_CITE_NUM = re.compile(r"\s*(?:,|\band\b|&)\s*(\d+[a-z]?)\b")
# Paragraph, range and point suffixes: "Art. 28(3)", "Art. 26(5)-(12)", "Art. 5(1)(f)", "Annex III, point 5(c)".
RE_CITE_SUFFIX = re.compile(r"(?:\(\d+[a-z]?\))*(?:\s*[-–]\s*\(\d+[a-z]?\))?(?:\([a-z]{1,4}\))*"
                            r"(?:,?\s*points?\s+\d+(?:\([a-z]{1,4}\))*)?")


@dataclass
class Answer:
    question: str
    text: str
    refused: bool
    citations: list[str]                 # unit ids cited, e.g. ["AIA-Art-27"]
    ungrounded_citations: list[str]      # cited but not in the retrieved context
    nonexistent_citations: list[str] = field(default_factory=list)  # cited but not in the regulation at all
    hits: list[Hit] = field(default_factory=list)
    model: str = ""
    prompt_tokens: int = 0
    completion_tokens: int = 0
    latency_s: float = 0.0

    @property
    def retrieved_units(self) -> list[str]:
        seen: list[str] = []
        for h in self.hits:
            if h.chunk["unit_id"] not in seen:
                seen.append(h.chunk["unit_id"])
        return seen


def build_context(hits: list[Hit]) -> str:
    """Group retrieved chunks by legal unit so each source label appears once."""
    by_unit: dict[str, list[dict]] = {}
    for h in hits:
        by_unit.setdefault(h.chunk["unit_id"], []).append(h.chunk)
    blocks = []
    for chunks in by_unit.values():
        c0 = chunks[0]
        title = f" ({c0['title']})" if c0["title"] else ""
        body = "\n".join(c["text"].split("\n", 1)[1] if "\n" in c["text"] else c["text"]
                         for c in sorted(chunks, key=lambda c: c["chunk_id"]))
        blocks.append(f"[SOURCE: {c0['label']}]{title}\n{body}")
    return "\n\n".join(blocks)


def iter_citations(text: str):
    """Yield (start, end, unit_id) for every citation, including follow-ons that inherit the
    regulation: "AI Act Art. 16(a), Art. 26(5)" -> AIA-Art-16 and AIA-Art-26."""
    for m in RE_CITE.finditer(text):
        src = SHORT_TO_KEY[m.group(1).lower()]
        uid = f"{src}-Art-{m.group(3).lower()}" if m.group(3) else f"{src}-Annex-{m.group(5).upper()}"
        end = RE_CITE_SUFFIX.match(text, m.end()).end()
        yield m.start(), end, uid
        plural = (m.group(2) or "").lower() == "articles"
        while True:
            if c := RE_CITE_CONT.match(text, end):
                plural = (c.group(1) or "").lower() == "articles"
                uid = f"{src}-Art-{c.group(2).lower()}" if c.group(2) else f"{src}-Annex-{c.group(4).upper()}"
                start = c.start(1) if c.group(1) else c.start(3)
            elif plural and (c := RE_CITE_NUM.match(text, end)):
                uid, start = f"{src}-Art-{c.group(1).lower()}", c.start(1)
            else:
                break
            end = RE_CITE_SUFFIX.match(text, c.end()).end()
            yield start, end, uid


def parse_citations(text: str) -> list[str]:
    ids: list[str] = []
    for _, _, uid in iter_citations(text):
        if uid not in ids:
            ids.append(uid)
    return ids


# A refusal declines to answer from the texts. A decline can be paraphrased ("I can't find
# the X in the provided AI Act or DORA texts"), so the decline is a pattern, not a fixed
# sentence - but a decline followed by figures (percentages, money, millions) means the
# model answered from memory after refusing, which is a failure, not a refusal.
RE_DECLINE = re.compile(r"can(?:'t|not)\s+find\b[^.]*\b(?:texts?|sources?)", re.I)
RE_OUTSIDE_FACT = re.compile(r"\d+\s*%|eur\s*[\d,.]+|€\s*[\d,.]+|\b(?:million|billion)\b", re.I)


def is_refusal(text: str) -> bool:
    t = text.lower().replace("’", "'")
    return bool(RE_DECLINE.search(t)) and not RE_OUTSIDE_FACT.search(t)


def prepare(question: str, index: Index, k: int = config.TOP_K):
    """Retrieve and build the prompt. Shared by answer() and the streaming UI."""
    hits = index.search(question, k=k)
    context = build_context(hits) if hits else "(no sources found)"
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": f"Sources:\n\n{context}\n\nQuestion: {question}"},
    ]
    return hits, messages


def finalise(question: str, text: str, hits: list[Hit], res, known_units: set[str] | None = None) -> Answer:
    """Parse citations/grounding once the full text exists. `known_units` (every unit id in the
    corpus) separates invented Articles from real ones that just weren't retrieved."""
    cites = parse_citations(text)
    retrieved = {h.chunk["unit_id"] for h in hits}
    return Answer(
        question=question,
        text=text.strip(),
        refused=is_refusal(text),
        citations=cites,
        ungrounded_citations=[c for c in cites if c not in retrieved],
        nonexistent_citations=[c for c in cites if known_units is not None and c not in known_units],
        hits=hits,
        model=res.model,
        prompt_tokens=res.prompt_tokens,
        completion_tokens=res.completion_tokens,
        latency_s=res.latency_s,
    )


def answer(question: str, index: Index, client, model: str | None = None, k: int = config.TOP_K) -> Answer:
    hits, messages = prepare(question, index, k=k)
    res = client.chat(messages, model=model)
    return finalise(question, res.text, hits, res, known_units=index.unit_ids)


def answer_stream(question: str, index: Index, client, model: str | None = None, k: int = config.TOP_K):
    """Streaming variant for the UI: returns (hits, delta_generator, result_holder)."""
    hits, messages = prepare(question, index, k=k)
    gen, result = client.chat_stream(messages, model=model)
    return hits, gen, result


def source_url(chunk: dict) -> str:
    return SOURCES[chunk["source"]]["eli_url"]
