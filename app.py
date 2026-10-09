"""Streamlit UI.  Run:  streamlit run app.py"""
import streamlit as st

from regcopilot import config
from regcopilot.index import Index
from regcopilot.llm import get_client
from regcopilot.rag import answer, source_url

st.set_page_config(page_title="EU AI Act & DORA Copilot", page_icon="🇪🇺", layout="wide")

MODES = {
    "Mistral API (La Plateforme)": ("mistral", config.MISTRAL_CHAT_MODEL),
    "Sovereign mode: local open-weight model (Ollama)": ("ollama", config.OLLAMA_CHAT_MODEL),
}

with st.sidebar:
    st.header("Settings")
    mode = st.radio("Where should the model run?", list(MODES))
    backend, default_model = MODES[mode]
    model = st.text_input("Model", default_model)
    available = Index.available()
    preferred = backend if backend in available else "none"
    embed = st.selectbox("Retrieval index", available or ["none"],
                         index=(available.index(preferred) if preferred in available else 0),
                         help="'none' = keyword search (BM25) only. Build embeddings with `python -m regcopilot index`.")
    k = st.slider("Sources to retrieve", 2, 12, config.TOP_K)
    if backend == "ollama":
        st.success("No question or document leaves this machine.")
    st.caption("Corpus: Regulation (EU) 2024/1689 (AI Act) and Regulation (EU) 2022/2554 (DORA), from EUR-Lex. "
               "Research prototype, not legal advice.")


@st.cache_resource
def _index(name: str) -> Index:
    return Index.load(name)


@st.cache_resource
def _client(name: str):
    return get_client(name)


st.title("EU AI Act & DORA Copilot")
st.write("Ask about obligations under the EU AI Act and DORA. Every answer is grounded in, and cites, the regulation text.")

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
if question:
    try:
        with st.spinner("Retrieving and answering..."):
            a = answer(question, _index(embed), _client(backend), model=model, k=k)
        st.session_state.history.append(a)
    except Exception as e:  # show setup problems clearly in the UI
        st.error(f"{type(e).__name__}: {e}")

for a in reversed(st.session_state.history):
    with st.chat_message("user"):
        st.write(a.question)
    with st.chat_message("assistant"):
        st.markdown(a.text)
        if a.ungrounded_citations:
            st.warning("Cited but not in the retrieved sources: " + ", ".join(a.ungrounded_citations))
        st.caption(f"{a.model} · {a.prompt_tokens}+{a.completion_tokens} tokens · {a.latency_s:.1f}s")
        with st.expander(f"Sources ({len(a.retrieved_units)})"):
            for h in a.hits:
                c = h.chunk
                cited = "✅ cited" if c["unit_id"] in a.citations else ""
                st.markdown(f"**{c['label']}**{(' — ' + c['title']) if c['title'] else ''} {cited}  \n"
                            f"[Open on EUR-Lex]({source_url(c)})")
                st.text(c["text"][:1200])
