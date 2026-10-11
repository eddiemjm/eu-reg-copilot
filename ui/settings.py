"""Sidebar settings, shared by every page through st.session_state["settings"]."""
from __future__ import annotations

import json
import os
from dataclasses import dataclass

import requests
import streamlit as st

from regcopilot import config
from regcopilot.index import Index

MODES = {"La Plateforme": "mistral", "Sovereign": "ollama"}
if os.getenv("REGCOPILOT_UI_MOCK"):  # offline stand-in, used by the UI smoke test
    MODES["Offline mock"] = "mock"

MODE_ICONS = {"mistral": ":material/cloud:", "ollama": ":material/lock:", "mock": ":material/science:"}
HOSTED_MODELS = ["mistral-small-latest", "mistral-large-latest"]
INDEX_NAMES = {"mistral": "hybrid · mistral-embed", "ollama": "hybrid · local embeddings",
               "mock": "hybrid · mock embeddings", "none": "keyword only (BM25)"}


@dataclass
class Settings:
    backend: str
    model: str
    embed: str
    k: int
    local: bool
    mode_label: str
    has_corpus: bool
    blocker: str = ""          # why asking is disabled, as Markdown; empty when ready
    ollama_models: tuple = ()  # chat models installed in Ollama, when it is running

    @property
    def can_ask(self) -> bool:
        return not self.blocker


@st.cache_data(ttl=30, show_spinner=False)
def ollama_models() -> list[str] | None:
    """Installed Ollama models, or None if Ollama isn't reachable."""
    try:
        r = requests.get(f"{config.OLLAMA_BASE_URL.rstrip('/')}/api/tags", timeout=0.5)
        r.raise_for_status()
        return [m["name"] for m in r.json().get("models", [])]
    except Exception:
        return None


def _installed(name: str, models: list[str]) -> bool:
    return any(m == name or m.split(":")[0] == name for m in models)


@st.cache_data(show_spinner=False)
def corpus_stats(mtime: float) -> dict:
    chunks = json.loads(config.CHUNKS_PATH.read_text())
    return {"chunks": len(chunks), "units": len({c["unit_id"] for c in chunks})}


def _blocker(backend: str, model: str, embed: str, has_corpus: bool, installed: list[str] | None) -> str:
    if not has_corpus:
        return ("**The corpus hasn't been built yet.** Run these once from the project folder:\n\n"
                "```bash\npython -m regcopilot ingest\npython -m regcopilot index --embed mistral\n```")
    if (backend == "mistral" or embed == "mistral") and not config.MISTRAL_API_KEY:
        return "**`MISTRAL_API_KEY` is not set.** Copy `.env.example` to `.env`, add your key, and reload."
    if backend == "ollama" or embed == "ollama":
        if installed is None:
            return "**Ollama isn't running.** Start it with `ollama serve`, then reload."
        needed = ([model] if backend == "ollama" else []) + ([config.OLLAMA_EMBED_MODEL] if embed == "ollama" else [])
        for name in needed:
            if not _installed(name, installed):
                return f"**`{name}` isn't installed in Ollama.** Run `ollama pull {name}`."
    return ""


def sidebar() -> Settings:
    with st.sidebar:
        st.markdown("#### Where the model runs")
        modes = list(MODES)
        mode = st.segmented_control("Where the model runs", modes, default=modes[0], key="mode",
                                    format_func=lambda m: f"{MODE_ICONS[MODES[m]]} {m}",
                                    label_visibility="collapsed") or modes[0]
        backend = MODES[mode]
        installed = ollama_models() if backend == "ollama" else None
        if backend == "mistral":
            st.caption("Your question and the retrieved passages are sent to `api.mistral.ai`.")
        elif backend == "ollama":
            st.caption("Questions, passages and embeddings stay on this machine.")
        else:
            st.caption("Deterministic offline stand-in for tests.")

        if backend == "mistral":
            choices = HOSTED_MODELS
        elif backend == "ollama":
            local = [m for m in installed or [] if "embed" not in m]
            choices = local or [config.OLLAMA_CHAT_MODEL]
        else:
            choices = ["mock-extractive"]
        model = st.selectbox("Model", choices, key=f"model-{backend}")

        available = Index.available()
        # A hosted embedding index would send every query to the API, so sovereign mode can't use it.
        options = [i for i in available if not (backend == "ollama" and i == "mistral")] or ["none"]
        preferred = backend if backend in options else "none"
        with st.expander("Retrieval", expanded=False):
            embed = st.selectbox("Index", options, index=options.index(preferred) if preferred in options else 0,
                                 format_func=lambda i: INDEX_NAMES.get(i, i), key=f"embed-{backend}",
                                 help="Hybrid fuses BM25 with dense embeddings (RRF). "
                                      "Build an index with `python -m regcopilot index --embed mistral`.")
            k = st.slider("Sources to retrieve", 2, 12, config.TOP_K, key="k")
        if backend != "mock" and backend not in available and config.CHUNKS_PATH.exists():
            st.caption(f"No `{backend}` embedding index yet, so retrieval is keyword-only.")

        has_corpus = config.CHUNKS_PATH.exists()
        blocker = _blocker(backend, model, embed, has_corpus, installed)
        if has_corpus:
            stats = corpus_stats(config.CHUNKS_PATH.stat().st_mtime)
            st.caption(f"Corpus · {stats['units']} Articles & Annexes · {stats['chunks']} chunks · "
                       "AI Act + DORA, from EUR-Lex")
        st.caption("Research prototype, not legal advice.")
        if st.session_state.get("history"):
            if st.button("Clear conversation", icon=":material/delete_sweep:", type="tertiary"):
                st.session_state.history = []
                st.rerun()

    label = {"mistral": "☁ La Plateforme", "ollama": "🔒 On this machine", "mock": "🧪 Offline mock"}[backend]
    return Settings(backend, model, embed, k, backend == "ollama", label, has_corpus, blocker,
                    tuple(m for m in installed or [] if "embed" not in m))
