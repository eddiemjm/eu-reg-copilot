"""HTML fragments for the UI. Pure functions (no Streamlit import) so they can be unit-tested.

The key one is `citation_html`: it turns the citations in a model answer into status pills, so a
reader sees at a glance which claims are backed by retrieved text and which are not.
"""
from __future__ import annotations

import html
import re

from regcopilot.rag import iter_citations
from regcopilot.sources import SOURCES

# A [...] block left holding only pills and separators after substitution: drop its brackets.
RE_PILL_BLOCK = re.compile(r'\[((?:\s|,|;|\band\b|<span class="rc-pill [a-z]+" title="[^"]*">[^<]*</span>)+)\]')

STATUS = {
    "ok": ("✓", "Verified: this Article was among the retrieved sources."),
    "missing": ("!", "Not retrieved: the model cited an Article it was not given."),
    "invented": ("✕", "Does not exist in the regulation."),
}


def unit_label(uid: str) -> str:
    """Readable label for a unit id like 'DORA-Art-28' or 'AIA-Annex-III'."""
    src, _, rest = uid.partition("-")
    if src not in SOURCES:
        return uid
    kind, _, num = rest.partition("-")
    return f"{SOURCES[src]['short']} {'Annex' if kind == 'Annex' else 'Art.'} {num}"


def status_of(uid: str, retrieved: set[str], known: set[str] | None = None) -> str:
    if known is not None and uid not in known:
        return "invented"
    return "ok" if uid in retrieved else "missing"


def _tip(uid: str, chunks_by_unit: dict) -> str:
    c = chunks_by_unit.get(uid)
    if not c:
        return unit_label(uid)
    body = c["text"].split("\n", 1)[-1].replace("\n", " ")
    head = f"{c['label']} — {c['title']}" if c["title"] else c["label"]
    return f"{head}\n{body[:180]}…"


def pill(label_html: str, uid: str, status: str, chunks_by_unit: dict | None = None) -> str:
    """`label_html` must already be escaped."""
    icon, desc = STATUS[status]
    # Newlines as entities: a blank line inside the tag would make Markdown split the HTML apart.
    tip = html.escape(f"{desc}\n\n{_tip(uid, chunks_by_unit or {})}", quote=True).replace("\n", "&#10;")
    return f'<span class="rc-pill {status}" title="{tip}">{icon} {label_html}</span>'


def citation_html(text: str, retrieved: set[str], known: set[str] | None = None,
                  chunks_by_unit: dict | None = None) -> str:
    """Escape a model answer and replace its citations with status pills. The result is Markdown
    with inline HTML, for st.markdown(..., unsafe_allow_html=True)."""
    chunks_by_unit = chunks_by_unit or {}
    # Escape HTML from the model; "$" would otherwise start LaTeX in Streamlit's Markdown.
    esc = html.escape(text, quote=False).replace("$", "\\$")
    out, last = [], 0
    for start, end, uid in iter_citations(esc):
        out += [esc[last:start], pill(esc[start:end], uid, status_of(uid, retrieved, known), chunks_by_unit)]
        last = end
    joined = "".join(out) + esc[last:]
    return RE_PILL_BLOCK.sub(lambda m: m.group(1).strip() if "rc-pill" in m.group(1) else m.group(0), joined)


def cost(model: str, prompt_tokens: int, completion_tokens: int, prices: dict) -> float | None:
    p = prices.get(model) or {}
    rate_in, rate_out = p.get("input_per_m") or 0, p.get("output_per_m") or 0
    if not (rate_in or rate_out):
        return None
    return prompt_tokens * rate_in / 1e6 + completion_tokens * rate_out / 1e6
