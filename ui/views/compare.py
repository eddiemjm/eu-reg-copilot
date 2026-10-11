"""Compare: one question, one set of retrieved sources, two models side by side."""
from __future__ import annotations

import html

import streamlit as st

from regcopilot import config
from regcopilot.rag import finalise, prepare
from ui import components as c
from ui import render
from ui.components import EXAMPLES
from ui.settings import HOSTED_MODELS
from ui.theme import SERIES

s = st.session_state["settings"]

st.markdown("#### Same question, same sources, two models")
st.caption("Retrieval runs once, so any difference comes from the model alone: what it cites, "
           "whether those citations were in its sources, how fast, and at what cost.")

if s.backend == "mistral":
    options = HOSTED_MODELS
elif s.backend == "ollama":
    options = list(s.ollama_models)
else:
    options = ["mock-a", "mock-b"]

if not s.can_ask:
    st.warning(s.blocker, icon=":material/build:")
    st.stop()
if len(options) < 2:
    st.info("Compare needs two local models. Pull another one, for example `ollama pull mistral-small`, "
            "or switch to La Plateforme.", icon=":material/compare_arrows:")
    st.stop()


def _example() -> None:
    if st.session_state.get("cmp-ex"):
        st.session_state["cmp-q"] = st.session_state["cmp-ex"]


st.selectbox("Start from an example", [""] + [q for _, q in EXAMPLES], key="cmp-ex", on_change=_example,
             format_func=lambda q: q or "Choose an example…")
with st.form("compare"):
    q = st.text_area("Question", key="cmp-q", height=80, placeholder="Ask about the AI Act or DORA…")
    a_col, b_col, go_col = st.columns([0.4, 0.4, 0.2], vertical_alignment="bottom")
    model_a = a_col.selectbox("Model A", options, index=0)
    model_b = b_col.selectbox("Model B", options, index=1)
    submitted = go_col.form_submit_button("Compare", type="primary", width="stretch")


def _verdict(answers) -> str:
    a, b = answers
    lines = []
    for x in answers:
        if x.nonexistent_citations:
            lines.append(f"<b>{x.model}</b> cited an Article that doesn't exist: "
                         + ", ".join(render.unit_label(u) for u in x.nonexistent_citations) + ".")
        missing = [u for u in x.ungrounded_citations if u not in x.nonexistent_citations]
        if missing:
            lines.append(f"<b>{x.model}</b> cited {len(missing)} Article{'s' if len(missing) > 1 else ''} "
                         f"it wasn't given: " + ", ".join(render.unit_label(u) for u in missing) + ".")
    if not lines:
        both = all(x.citations for x in answers)
        lines.append("Every citation in both answers is backed by the retrieved sources." if both
                     else "Neither answer cites anything it wasn't given.")
    fast, slow = sorted(answers, key=lambda x: x.latency_s)
    if fast.latency_s > 0:
        lines.append(f"<b>{fast.model}</b> answered {slow.latency_s / fast.latency_s:.1f}× faster.")
    prices = config.load_prices()
    costs = {x.model: render.cost(x.model, x.prompt_tokens, x.completion_tokens, prices) for x in answers}
    if all(costs.values()) and costs[a.model] != costs[b.model]:
        cheap, dear = sorted(answers, key=lambda x: costs[x.model])
        lines.append(f"<b>{cheap.model}</b> cost {costs[dear.model] / costs[cheap.model]:.1f}× less.")
    return '<div class="rc-verdict">' + " ".join(lines) + "</div>"


def _column_head(model: str) -> str:
    dot = SERIES.get(model, "#9A9AA6")
    return f'<div class="rc-col-head"><span class="rc-dot" style="background:{dot}"></span><b>{html.escape(model)}</b></div>'


if submitted and q.strip():
    index = c.get_index(s.embed)
    hits, messages = prepare(q.strip(), index, k=s.k)
    answers = []
    for col, model in zip(st.columns(2, gap="large"), (model_a, model_b)):
        with col:
            st.markdown(_column_head(model), unsafe_allow_html=True)
            try:
                gen, res = c.get_llm(s.backend).chat_stream(messages, model=model)
                body = st.empty()
                text = body.write_stream(gen)
                a = finalise(q.strip(), text, hits, res, known_units=index.unit_ids)
                body.markdown(c.answer_html(a, s.embed), unsafe_allow_html=True)
                st.markdown(c.badge_html(a), unsafe_allow_html=True)
                answers.append(a)
            except Exception as e:
                st.error(f"{type(e).__name__}: {e}")
    if len(answers) == 2:
        st.session_state["cmp-result"] = answers
        st.markdown(_verdict(answers), unsafe_allow_html=True)
        c.evidence(hits, set(answers[0].citations) | set(answers[1].citations), "cmp-live")
elif st.session_state.get("cmp-result"):
    answers = st.session_state["cmp-result"]
    st.markdown(f"**{html.escape(answers[0].question)}**")
    for col, a in zip(st.columns(2, gap="large"), answers):
        with col:
            st.markdown(_column_head(a.model), unsafe_allow_html=True)
            c.show_answer(a, s.embed)
    st.markdown(_verdict(answers), unsafe_allow_html=True)
    c.evidence(answers[0].hits, set(answers[0].citations) | set(answers[1].citations), "cmp")
