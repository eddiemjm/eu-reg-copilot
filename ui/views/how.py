"""How it works: the pipeline, the design decisions, where the texts came from, and a retrieval inspector."""
from __future__ import annotations

import json
from collections import Counter

import pandas as pd
import streamlit as st

from regcopilot import config
from regcopilot.sources import SOURCES
from ui import components as c
from ui.settings import INDEX_NAMES

s = st.session_state["settings"]

st.markdown("#### The pipeline")
node = 'style="rounded,filled" fillcolor="#17171B" color="#3A3A44" fontcolor="#ECECEF" fontname="Helvetica" fontsize=13'
hot = 'style="rounded,filled" fillcolor="#2A1408" color="#FA500F" fontcolor="#ECECEF" fontname="Helvetica" fontsize=13'
st.graphviz_chart(f"""
digraph {{
  rankdir=LR; bgcolor="transparent"; nodesep=0.3; ranksep=0.4;
  node [shape=box margin="0.18,0.08" {node}]; edge [color="#6B6B76" arrowsize=0.6 penwidth=1.2];
  texts  [label="Official texts\\nEUR-Lex via Cellar\\nparsed into Articles"];
  q      [label="Question" {hot}];
  bm25   [label="BM25\\nexact terms"];
  dense  [label="Embeddings\\nparaphrases"];
  fuse   [label="RRF fusion\\n+ named Articles"];
  model  [label="Mistral model\\ngrounded prompt" {hot}];
  check  [label="Citation check\\ncited ⊆ retrieved ⊆ corpus"];
  texts -> bm25; texts -> dense; q -> bm25 [style=dashed]; q -> dense [style=dashed];
  bm25 -> fuse; dense -> fuse; fuse -> model -> check;
}}
""", width="stretch")

DECISIONS = [
    ("Structure over sliding windows", "Lawyers cite Articles, so Articles are the retrieval unit. That makes every "
     "citation checkable and lets the eval score retrieval against a known right answer."),
    ("Hybrid retrieval", "BM25 catches exact terms (\"register of information\", \"TLPT\"); embeddings catch "
     "paraphrases (\"can my boss read my mood?\"). Reciprocal Rank Fusion combines them without tuned weights."),
    ("Refusal is a feature", "A fixed refusal sentence makes \"I don't know\" measurable. Out-of-scope questions "
     "check the model doesn't answer GDPR questions from AI Act text."),
    ("Verify after generation", "Each citation is checked against the retrieved passages and the corpus, so an "
     "Article cited from memory, or one that doesn't exist, is flagged rather than trusted."),
    ("Deployment choice", "The same pipeline runs on La Plateforme or on open-weight models via Ollama, with "
     "nothing leaving the machine. Plain HTTP, no SDK, so the calls are easy to audit."),
    ("Measured, not assumed", "35 golden questions score retrieval, citations, grounding, refusals, a judge, "
     "latency and cost for every model and index."),
]
for row in range(0, len(DECISIONS), 3):
    for col, (title, body) in zip(st.columns(3), DECISIONS[row:row + 3]):
        col.markdown(f'<div class="rc-card"><h4>{title}</h4><p>{body}</p></div>', unsafe_allow_html=True)
    st.write("")

st.markdown("#### The corpus")
if not config.CHUNKS_PATH.exists():
    st.info("Not built yet: run `python -m regcopilot ingest`.")
else:
    chunks = json.loads(config.CHUNKS_PATH.read_text(encoding="utf-8"))
    manifest_path = config.RAW_DIR / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8")) if manifest_path.exists() else {}
    cols = st.columns(len(SOURCES))
    for col, (key, src) in zip(cols, SOURCES.items()):
        units = Counter(c_["kind"] for c_ in {c_["unit_id"]: c_ for c_ in chunks if c_["source"] == key}.values())
        n_chunks = sum(1 for c_ in chunks if c_["source"] == key)
        m = manifest.get(key, {})
        prov = (f"Fetched {m['fetched_at'][:10]} via `{m['strategy']}` · SHA-256 `{m['sha256'][:12]}…`"
                if m.get("sha256") else "Provenance not recorded")
        with col:
            st.markdown(f"**{src['short']}** · {src['title']}")
            a, b, cc = st.columns(3)
            a.metric("Articles", units.get("article", 0))
            b.metric("Annexes", units.get("annex", 0))
            cc.metric("Chunks", n_chunks)
            st.caption(prov)

@st.cache_data(show_spinner="Searching…", max_entries=50)
def _search(embed: str, query: str):
    """Cached, so revisiting the page doesn't re-embed the same query."""
    return c.get_index(embed).search(query, k=10)


st.markdown("#### Retrieval inspector")
st.caption("See how BM25 and dense ranks fuse for any query, before a model is involved. "
           f"Index: {INDEX_NAMES.get(s.embed, s.embed)}.")
query = st.text_input("Query", "Can my employer use AI to read my mood at work?", key="insp-q")
if query.strip() and config.CHUNKS_PATH.exists():
    if s.embed == "mistral" and not config.MISTRAL_API_KEY:
        st.info("The hybrid index needs `MISTRAL_API_KEY` to embed the query. Showing BM25 only.")
        index = c.get_index("none")
    else:
        index = c.get_index(s.embed)
    try:
        hits = _search(index.embed_backend, query)
        st.dataframe(pd.DataFrame([{
            "fused rank": i + 1, "unit": h.chunk["label"], "title": h.chunk["title"],
            "chunk": h.chunk["chunk_id"].split("#")[1], "BM25 rank": None if h.bm25_rank is None else h.bm25_rank + 1,
            "dense rank": None if h.dense_rank is None else h.dense_rank + 1,
            "RRF score": round(h.score, 4), "named": h.score >= 1.0} for i, h in enumerate(hits)]),
            hide_index=True, width="stretch",
            column_config={"named": st.column_config.CheckboxColumn("named in query")})
        st.caption("RRF score = Σ 1 / (60 + rank) over both rankings. A blank rank means that retriever didn't "
                   "put the chunk in its top 50. Articles named in the query are pinned to the top.")
    except Exception as e:
        st.error(f"{type(e).__name__}: {e}")
