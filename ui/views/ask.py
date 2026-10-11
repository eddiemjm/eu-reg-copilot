"""Ask: chat with streamed answers, inline citation pills and the evidence beside each answer."""
from __future__ import annotations

import html
import time

import streamlit as st

from regcopilot.rag import answer_stream, finalise
from ui import components as c
from ui.components import EXAMPLES, TAG_COLOUR

s = st.session_state["settings"]
history = st.session_state.setdefault("history", [])


def _ask(q: str) -> None:
    st.session_state.pending = q


def _turn(a, key: str) -> None:
    with st.chat_message("user", avatar=":material/person:"):
        st.markdown(html.escape(a.question))
    with st.chat_message("assistant", avatar=":material/balance:"):
        left, right = st.columns([0.62, 0.38], gap="large")
        with left:
            c.show_answer(a, s.embed)
        with right:
            c.evidence(a.hits, set(a.citations), key)


question = st.chat_input("Ask about the AI Act or DORA…" if s.can_ask else "Finish setup to start asking",
                         disabled=not s.can_ask)
question = question or st.session_state.pop("pending", None)

if not s.can_ask:
    st.warning(s.blocker, icon=":material/build:")

if not history and not question:
    st.markdown('<div class="rc-hint">Try one of these, or ask your own below. Every citation in the answer '
                'is checked against the passages the model was given.</div>', unsafe_allow_html=True)
    for row in range(0, len(EXAMPLES), 3):
        for col, (i, (tag, q)) in zip(st.columns(3), enumerate(EXAMPLES[row:row + 3], start=row)):
            with col:
                st.button(f":{TAG_COLOUR[tag]}-badge[{tag}]  {q}", key=f"ex-{i}", on_click=_ask, args=(q,),
                          disabled=not s.can_ask, width="stretch")

for i, a in enumerate(history):
    _turn(a, f"h{i}")

if question and s.can_ask:
    with st.chat_message("user", avatar=":material/person:"):
        st.markdown(html.escape(question))
    with st.chat_message("assistant", avatar=":material/balance:"):
        left, right = st.columns([0.62, 0.38], gap="large")
        a = None
        with left:
            try:
                t0 = time.perf_counter()
                with st.status("Searching the AI Act and DORA…") as status:
                    index = c.get_index(s.embed)
                    hits, gen, res = answer_stream(question, index, c.get_llm(s.backend), model=s.model, k=s.k)
                    n_units = len({h.chunk["unit_id"] for h in hits})
                    status.update(label=f"Retrieved {n_units} Articles in {time.perf_counter() - t0:.2f}s",
                                  state="complete", expanded=False)
                preview = right.empty()
                preview.markdown(c.preview_html(hits), unsafe_allow_html=True)
                body = st.empty()
                text = body.write_stream(gen)
                a = finalise(question, text, hits, res, known_units=index.unit_ids)
                body.markdown(c.answer_html(a, s.embed), unsafe_allow_html=True)
                st.markdown(c.badge_html(a), unsafe_allow_html=True)
            except Exception as e:  # show setup problems clearly instead of a stack trace
                st.error(f"{type(e).__name__}: {e}")
        if a is not None:
            preview.empty()
            with right:
                c.evidence(a.hits, set(a.citations), f"h{len(history)}")
            history.append(a)
