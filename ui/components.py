"""Streamlit pieces shared by the Ask and Compare pages."""
from __future__ import annotations

import html

import streamlit as st

from regcopilot import config
from regcopilot.index import Hit, Index
from regcopilot.llm import get_client
from regcopilot.rag import Answer, source_url
from ui import render, results

EXAMPLES = [
    ("AI Act", "Does an insurer using AI to price life insurance need a fundamental rights impact assessment?"),
    ("DORA", "How often must financial entities run threat-led penetration testing?"),
    ("Both", "A bank uses a third-party AI model for credit scoring. What do the AI Act and DORA each require?"),
    ("AI Act", "Can an employer use emotion recognition to monitor staff?"),
    ("DORA", "What must contracts with ICT third-party service providers include?"),
    ("AI Act", "When does a general-purpose AI model count as having systemic risk?"),
]
TAG_COLOUR = {"AI Act": "orange", "DORA": "blue", "Both": "violet"}


@st.cache_resource(show_spinner="Loading the index…")
def get_index(name: str) -> Index:
    return Index.load(name)


@st.cache_resource
def get_llm(backend: str):
    return get_client(backend)


@st.cache_resource
def chunks_by_unit(name: str) -> dict:
    """First chunk of every legal unit, for citation tooltips."""
    out: dict = {}
    for c in get_index(name).chunks:
        out.setdefault(c["unit_id"], c)
    return out


def plain_md(text: str) -> str:
    """Regulation text as safe Markdown, keeping its line breaks."""
    return html.escape(text, quote=False).replace("$", "\\$").replace("\n", "  \n")


def answer_html(a: Answer, embed: str) -> str:
    return render.citation_html(a.text, set(a.retrieved_units), get_index(embed).unit_ids, chunks_by_unit(embed))


def badge_html(a: Answer) -> str:
    n, bad, invented = len(a.citations), len(a.ungrounded_citations), len(a.nonexistent_citations)
    if a.refused:
        lead = ("none", "Declined: no basis in the retrieved text")
    elif not n:
        lead = ("none", "No citations")
    elif invented:
        lead = ("bad", f"✕ {invented} cited Article{'s' if invented > 1 else ''} doesn't exist")
    elif bad:
        lead = ("warn", f"! {n - bad}/{n} citations verified")
    else:
        lead = ("ok", f"✓ {n}/{n} citations verified")
    parts = [a.model, f"{len(a.retrieved_units)} sources", f"{a.latency_s:.1f}s",
             f"{a.prompt_tokens:,}+{a.completion_tokens:,} tokens"]
    c = render.cost(a.model, a.prompt_tokens, a.completion_tokens, config.load_prices())
    if c is not None:
        parts.append(f"${c:.4f}")
    return (f'<div class="rc-badge"><span class="lead {lead[0]}">{lead[1]}</span>'
            + "".join(f"<span>{html.escape(p)}</span>" for p in parts) + "</div>")


def show_answer(a: Answer, embed: str) -> None:
    st.markdown(answer_html(a, embed), unsafe_allow_html=True)
    st.markdown(badge_html(a), unsafe_allow_html=True)


def preview_html(hits: list[Hit]) -> str:
    """Lightweight list of sources shown while the answer streams."""
    seen = list(dict.fromkeys(h.chunk["label"] for h in hits))
    rows = "".join(f'<div class="rc-evi-meta">{html.escape(l)}</div>' for l in seen)
    return f'<div class="rc-evi-title">Evidence · {len(seen)} sources</div>{rows}'


def evidence(hits: list[Hit], cited: set[str], key: str) -> None:
    """One card per retrieved unit, cited ones marked, with the passage on demand."""
    units = list(dict.fromkeys(h.chunk["unit_id"] for h in hits))
    st.markdown(f'<div class="rc-evi-title">Evidence · {len(units)} sources retrieved</div>', unsafe_allow_html=True)
    for i, uid in enumerate(units):
        unit_hits = [h for h in hits if h.chunk["unit_id"] == uid]
        best, c = unit_hits[0], unit_hits[0].chunk
        is_cited = uid in cited
        ranks = []
        if best.bm25_rank is not None:
            ranks.append(f'<span class="rc-rank">BM25 #{best.bm25_rank + 1}</span>')
        if best.dense_rank is not None:
            ranks.append(f'<span class="rc-rank">dense #{best.dense_rank + 1}</span>')
        if best.score >= 1.0:
            ranks.append('<span class="rc-rank">named in question</span>')
        title = f" — {html.escape(c['title'])}" if c["title"] else ""
        with st.container(key=f"evi-{key}-{i}{'-cited' if is_cited else ''}"):
            st.markdown(
                f'<div class="rc-evi-head"><b>{html.escape(c["label"])}</b>{title}</div>'
                f'<div class="rc-evi-meta">{"".join(ranks)}'
                f'{"<span class=cited>● cited</span>" if is_cited else ""}</div>',
                unsafe_allow_html=True)
            with st.expander("Read passage"):
                for h in unit_hits:
                    st.markdown(plain_md(h.chunk["text"].split("\n", 1)[-1]))
                st.markdown(f"[Open the regulation on EUR-Lex ↗]({source_url(c)})")


def header_kpis(s) -> tuple[list[tuple[str, str]], str]:
    """Headline eval numbers for the selected model and index, from the latest matching run."""
    row = results.latest(results.load_summaries(config.RESULTS_DIR), s.model, s.embed)
    if not row:
        return [("citations", "Checked"), ("retrieval", "BM25 + dense")], ""
    fmt = lambda v, suffix="": "n/a" if v is None else f"{v:g}{suffix}"  # noqa: E731
    return ([("retrieval hit", fmt(row.get("hit@k %"), "%")), ("grounded", fmt(row.get("grounded %"), "%")),
             ("judge", fmt(row.get("judge (1-5)"), "/5")), ("p50 latency", fmt(row.get("p50 latency s"), "s"))],
            f"{s.model} · golden-set eval, {results.run_label(row['stamp'])}")
