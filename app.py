"""Streamlit UI.  Run:  streamlit run app.py"""
from __future__ import annotations

import json
import time

import streamlit as st

from regcopilot import config
from regcopilot.index import Index
from regcopilot.llm import get_client
from regcopilot.rag import answer_stream, finalise, source_url

st.set_page_config(page_title="EU AI Act & DORA Copilot", page_icon="🇪🇺", layout="wide")

st.markdown("""
<style>
.hero {
    background: linear-gradient(135deg, #0d2b8f 0%, #1a3aa8 55%, #0b1220 100%);
    border: 1px solid rgba(255, 204, 0, 0.25);
    border-radius: 14px;
    padding: 1.5rem 1.8rem 1.2rem;
    margin-bottom: 0.4rem;
}
.hero h1 { color: #ffffff; font-weight: 800; margin: 0 0 0.2rem; }
.hero p { color: #cdd7ee; margin: 0; }
.cards { display: flex; gap: 0.8rem; margin-top: 1rem; flex-wrap: wrap; }
.card {
    background: rgba(255, 255, 255, 0.06);
    border: 1px solid rgba(255, 255, 255, 0.12);
    border-radius: 10px;
    padding: 0.5rem 0.9rem;
    min-width: 130px;
}
.card b { display: block; font-size: 1.2rem; color: #ffcc00; }
.card span { color: #aab6d0; font-size: 0.78rem; }
.chip {
    display: inline-block;
    border-radius: 999px;
    padding: 0.12rem 0.6rem;
    margin: 0.15rem 0.25rem 0.15rem 0;
    font-size: 0.8rem;
}
.chip.ok { background: rgba(46, 204, 113, 0.15); border: 1px solid rgba(46, 204, 113, 0.5); color: #7dedae; }
.chip.bad { background: rgba(255, 180, 0, 0.12); border: 1px solid rgba(255, 180, 0, 0.5); color: #ffd166; }
</style>
""", unsafe_allow_html=True)

MODES = {
    "Mistral API (La Plateforme)": ("mistral", config.MISTRAL_CHAT_MODEL),
    "Sovereign mode: local open-weight model (Ollama)": ("ollama", config.OLLAMA_CHAT_MODEL),
}


def _card(value: str, label: str) -> str:
    return f'<div class="card"><b>{value}</b><span>{label}</span></div>'


def _corpus_stats():
    """(n_chunks, n_units) from the ingested corpus, or None if not built yet."""
    try:
        chunks = json.loads(config.CHUNKS_PATH.read_text())
        return len(chunks), len({c["unit_id"] for c in chunks})
    except Exception:
        return None, None


def _eval_row(model: str, embed: str) -> dict | None:
    """Most recent eval summary row for this model *and* retrieval index, from results/*_summary.md.
    Columns are looked up by header name, so adding a metric doesn't shift them."""
    try:
        for f in sorted(config.RESULTS_DIR.glob("*_summary.md"), reverse=True):
            lines = [l for l in f.read_text().splitlines() if l.startswith("|") and not l.startswith("|---")]
            if not lines:
                continue
            cell = lambda l: [c.strip() for c in l.strip().strip("|").split("|")]  # noqa: E731
            header = cell(lines[0])
            for line in lines[1:]:
                row = dict(zip(header, cell(line)))
                if row.get("model") == model and row.get("index") == embed:
                    return {"hit": row["hit@k %"], "judge": row["judge (1-5)"]}
        return None
    except Exception:
        return None


def _hero(model: str, embed: str):
    n_chunks, n_units = _corpus_stats()
    cards = ""
    if n_chunks:
        cards += _card(f"{n_chunks} chunks", f"AI Act + DORA · {n_units} articles & annexes")
    cards += _card("BM25 + dense", "hybrid RRF retrieval")
    ev = _eval_row(model, embed)
    if ev:
        cards += _card(f"{ev['hit']}% hit@k", f"LLM judge {ev['judge']}/5 · 35-Q golden set")
    else:
        cards += _card("Citation-checked", "grounding verified per answer")
    cards += _card("Hosted or sovereign", "Mistral API or fully on-device")
    st.markdown(
        f'<div class="hero"><h1>EU AI Act &amp; DORA Copilot</h1>'
        f'<p>Answers grounded in the official texts, with a citation to the exact Article for every claim.</p>'
        f'<div class="cards">{cards}</div></div>',
        unsafe_allow_html=True)


with st.sidebar:
    st.header("Settings")
    mode = st.radio("Where should the model run?", list(MODES))
    backend, default_model = MODES[mode]

    MODEL_CHOICES = {
    "mistral": ["mistral-small-latest", "mistral-large-latest"],
    "ollama": [config.OLLAMA_CHAT_MODEL],  # e.g. "mistral-nemo"
}
    model = st.selectbox("Model", MODEL_CHOICES[backend],index=MODEL_CHOICES[backend].index(default_model))
    available = Index.available()
    preferred = backend if backend in available else "none"
    embed = st.selectbox("Retrieval index", available or ["none"],
                         index=(available.index(preferred) if preferred in available else 0),
                         help="'none' = keyword search (BM25) only. Build embeddings with `python -m regcopilot index`.")
    k = st.slider("Sources to retrieve", 2, 12, config.TOP_K)
    if backend == "ollama":
        st.success("No question or document leaves this machine.")
    n_chunks, n_units = _corpus_stats()
    if n_chunks:
        st.caption(f"Corpus: {n_chunks} chunks · {n_units} articles/annexes · 2 regulations, from EUR-Lex via Cellar.")
    else:
        st.caption("Corpus not built yet — run `python -m regcopilot ingest`.")
    st.caption("Research prototype, not legal advice.")


@st.cache_resource
def _index(name: str) -> Index:
    return Index.load(name)


@st.cache_resource
def _client(name: str):
    return get_client(name)


def _unit_label(uid: str) -> str:
    """Readable label for a unit id like 'DORA-Art-28' or 'AIA-Annex-III'."""
    src = {"AIA": "AI Act", "DORA": "DORA"}.get(uid.split("-")[0])
    if not src:
        return uid
    kind, _, num = uid.split("-", 1)[1].partition("-")
    return f"{src} {'Annex' if kind == 'Annex' else 'Art.'} {num}"


def _cost(a) -> str:
    p = config.load_prices().get(a.model, {})
    rate_in = p.get("input_per_m", 0) or 0
    rate_out = p.get("output_per_m", 0) or 0
    if not (rate_in or rate_out):
        return ""
    return f"${a.prompt_tokens * rate_in / 1e6 + a.completion_tokens * rate_out / 1e6:.4f}"


def _answer_body(a) -> None:
    """Citation chips + caption + sources, rendered inside an assistant bubble."""
    if a.refused:
        st.info("No basis in the retrieved text — the assistant refused rather than guessing.")
    elif a.citations:
        labels = {h.chunk["unit_id"]: h.chunk["label"] for h in a.hits}
        chips = []
        for c in a.citations:
            if c in a.nonexistent_citations:
                chips.append(f'<span class="chip bad">does not exist · {_unit_label(c)}</span>')
            elif c in a.ungrounded_citations:
                chips.append(f'<span class="chip bad">not retrieved · {_unit_label(c)}</span>')
            else:
                chips.append(f'<span class="chip ok">✓ {labels.get(c, _unit_label(c))}</span>')
        st.markdown("".join(chips), unsafe_allow_html=True)
        if a.nonexistent_citations:
            st.error("Cited an Article or Annex that does not exist in the regulation: "
                     + ", ".join(_unit_label(c) for c in a.nonexistent_citations))
        not_retrieved = [c for c in a.ungrounded_citations if c not in a.nonexistent_citations]
        if not_retrieved:
            st.warning("Cited but not among the retrieved sources: "
                       + ", ".join(_unit_label(c) for c in not_retrieved))
    cap = f"{a.model} · {a.prompt_tokens:,}+{a.completion_tokens:,} tokens · {a.latency_s:.1f}s"
    cost = _cost(a)
    if cost:
        cap += f" · {cost}"
    st.caption(cap)
    with st.expander(f"Sources ({len(a.retrieved_units)})"):
        for h in a.hits:
            c = h.chunk
            ranks = []
            if h.bm25_rank is not None:
                ranks.append(f"BM25 #{h.bm25_rank}")
            if h.dense_rank is not None:
                ranks.append(f"dense #{h.dense_rank}")
            meta = " · ".join(ranks + [f"fused {h.score:.3f}"])
            cited = "✅ cited" if c["unit_id"] in a.citations else ""
            st.markdown(f"**{c['label']}**{(' — ' + c['title']) if c['title'] else ''} {cited}  \n"
                        f"`{meta}` · [Open on EUR-Lex]({source_url(c)})")
            st.text(c["text"][:800])


_hero(model, embed)

examples = [
    "Does an insurer using AI to price life insurance need a fundamental rights impact assessment?",
    "How often must financial entities run threat-led penetration testing under DORA?",
    "What must contracts with ICT third-party service providers include?",
    "When does a general-purpose AI model count as having systemic risk?",
]
cols = st.columns(len(examples))
clicked = None
for c, ex in zip(cols, examples):
    if c.button(ex, use_container_width=True):
        clicked = ex

if "history" not in st.session_state:
    st.session_state.history = []

question = st.chat_input("Ask a question about the AI Act or DORA") or clicked
appended = False
if question:
    try:
        with st.chat_message("user"):
            st.write(question)
        with st.chat_message("assistant"):
            t0 = time.perf_counter()
            status = st.status("Searching the AI Act and DORA...", expanded=False)
            with status:
                hits, gen, res = answer_stream(question, _index(embed), _client(backend),
                                               model=model, k=k)
                n_units = len({h.chunk["unit_id"] for h in hits})
                status.update(label=f"Retrieved {len(hits)} chunks from {n_units} articles "
                                    f"in {time.perf_counter() - t0:.2f}s",
                              state="complete", expanded=False)
            text = st.write_stream(gen)          # renders tokens live, returns full text
            a = finalise(question, text, hits, res,   # res is filled in now the stream is done
                         known_units=_index(embed).unit_ids)
            st.session_state.history.append(a)
            appended = True
            _answer_body(a)
    except Exception as e:  # show setup problems clearly in the UI
        st.error(f"{type(e).__name__}: {e}")

for a in reversed(st.session_state.history[:-1] if appended else st.session_state.history):
    with st.chat_message("user"):
        st.write(a.question)
    with st.chat_message("assistant"):
        st.markdown(a.text)
        _answer_body(a)
