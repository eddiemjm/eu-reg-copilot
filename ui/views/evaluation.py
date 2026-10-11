"""Evaluation: the golden-set results, how they changed across runs, and every question's answer."""
from __future__ import annotations

import json

import altair as alt
import pandas as pd
import streamlit as st

from regcopilot import config
from ui import components as c
from ui import render, results
from ui.settings import INDEX_NAMES
from ui.theme import SERIES

INK, MUTED, FAINT, GRID = "#ECECEF", "#9A9AA6", "#6B6B76", "rgba(255,255,255,0.07)"
PCT = ["hit@k %", "cited correctly %", "grounded %", "false refusals %", "OOS refused %"]

s = st.session_state["settings"]
rows = results.load_summaries(config.RESULTS_DIR)

st.markdown("#### Evaluation")
st.caption("35-question golden set (32 in-scope, 3 out-of-scope) with hand-written reference answers · "
           "k = 6 · judge: mistral-large-latest, scoring 1–5 against the reference")
if not rows:
    st.info("No eval results yet. Run:\n\n```bash\npython eval/run_eval.py --backend mistral "
            "--models mistral-small-latest,mistral-large-latest --judge mistral-large-latest\n```")
    st.stop()


def _cost_per_query(r: dict) -> float | None:
    # Older runs report cost 0.0 because prices.json was empty then, so treat 0 as unknown.
    return r["cost"] / r["n"] if r.get("cost") and r.get("n") else None


# ---------- headline metrics for one configuration ----------
configs = list(dict.fromkeys((r["model"], r["index"]) for r in reversed(rows)))
current = (s.model, s.embed) if (s.model, s.embed) in configs else configs[0]
model, index = st.selectbox("Configuration", configs, index=configs.index(current),
                            format_func=lambda mi: f"{mi[0]} · {INDEX_NAMES.get(mi[1], mi[1])}")
cur = results.latest(rows, model, index)
prev = results.latest(rows, model, index, before=cur["stamp"])


def _metric(col, label: str, key: str, unit: str = "", inverse: bool = False, getter=None) -> None:
    get = getter or (lambda r: r.get(key))
    v = get(cur)
    p = get(prev) if prev else None
    if v is None:
        col.metric(label, "n/a")
        return
    shown = f"${v:.4f}" if unit == "$" else f"{v:g}{unit}"
    delta = None
    if p is not None and v != p:
        delta = f"{v - p:+.4f}" if unit == "$" else f"{v - p:+.2f}{' pts' if unit == '%' else unit}"
    col.metric(label, shown, delta, delta_color="inverse" if inverse else "normal")


cols = st.columns(6)
_metric(cols[0], "Hit@6", "hit@k %", "%")
_metric(cols[1], "Cited correctly", "cited correctly %", "%")
_metric(cols[2], "Grounded", "grounded %", "%")
_metric(cols[3], "Judge (1–5)", "judge (1-5)")
_metric(cols[4], "p50 latency", "p50 latency s", "s", inverse=True)
_metric(cols[5], "Cost / query", "", "$", inverse=True, getter=_cost_per_query)
st.caption(f"Run {results.run_label(cur['stamp'])}"
           + (f" · deltas against {results.run_label(prev['stamp'])}" if prev else " · first run of this configuration"))

# ---------- the grounding story ----------
st.markdown("#### Finding: the larger model cited Articles it was never given")
st.markdown(
    "Grounded means every Article an answer cites was among the passages retrieved for it. "
    "`mistral-large` often added the AI Act's entry-into-force Article from memory, because the prompt "
    "asked when each obligation applies. Allowing dates only when they appear in the sources fixed most of it.")
hybrid = [r for r in rows if r["index"] == "mistral" and r["model"] in SERIES]
if hybrid:
    df = pd.DataFrame([{"run": results.run_label(r["stamp"]), "stamp": r["stamp"], "model": r["model"],
                        "grounded": r.get("grounded %"), "cited": r.get("cited correctly %"),
                        "judge": r.get("judge (1-5)")} for r in hybrid])
    order = list(dict.fromkeys(df.sort_values("stamp")["run"]))
    x = alt.X("run:N", sort=order, title=None, axis=alt.Axis(labelAngle=0))
    y = alt.Y("grounded:Q", scale=alt.Scale(domain=[0, 100]), title="Grounded %")
    colour = alt.Color("model:N", scale=alt.Scale(domain=list(SERIES), range=list(SERIES.values())),
                       legend=alt.Legend(orient="top", title=None, labelColor=INK, symbolType="square"))
    tip = [alt.Tooltip("model:N", title="Model"), alt.Tooltip("run:N", title="Run"),
           alt.Tooltip("grounded:Q", title="Grounded %"), alt.Tooltip("cited:Q", title="Cited correctly %"),
           alt.Tooltip("judge:Q", title="Judge")]
    base = alt.Chart(df).encode(x=x, y=y, color=colour)
    layers = [base.mark_line(strokeWidth=2),
              base.mark_point(filled=True, size=90, stroke="#17171B", strokeWidth=2, opacity=1).encode(tooltip=tip),
              # Direct labels on the latest run only; text stays in ink, the series colour lives on the marks.
              alt.Chart(df[df["stamp"] == df["stamp"].max()]).mark_text(align="left", dx=10, color=INK, fontSize=12)
              .encode(x=x, y=y, text=alt.Text("grounded:Q", format=".1f"))]
    notes = pd.DataFrame([{"run": results.run_label(st_), "note": n} for st_, n in results.NOTES.items()
                          if st_ in set(df["stamp"])])
    if not notes.empty:
        layers.insert(0, alt.Chart(notes).mark_rule(strokeDash=[4, 4], color=FAINT).encode(x=x))
        layers.append(alt.Chart(notes).mark_text(align="right", dx=-8, color=MUTED, fontSize=11)
                      .encode(x=x, y=alt.datum(30), text="note:N"))
    chart = (alt.layer(*layers).properties(height=280, background="transparent")
             .configure_view(stroke=None)
             .configure_axis(labelColor=MUTED, titleColor=MUTED, gridColor=GRID, domainColor=GRID, tickColor=GRID,
                             labelFontSize=12, titleFontSize=12, titleFontWeight="normal"))
    st.altair_chart(chart, theme=None)  # stretches by default
    with st.expander("Chart data"):
        st.dataframe(df.drop(columns="stamp"), hide_index=True, width="stretch")
    st.caption("Hybrid index. Earlier runs were scored with the old citation parser, which only read "
               "[square-bracket] citations.")

# ---------- latest run, side by side ----------
left, right = st.columns([0.55, 0.45], gap="large")
with left:
    st.markdown("#### Latest hybrid run")
    if hybrid:
        last = max(r["stamp"] for r in hybrid)
        latest_rows = [r for r in hybrid if r["stamp"] == last]
        metrics = PCT + ["MRR", "nonexistent cites", "judge (1-5)", "p50 latency s"]
        table = {r["model"]: [("n/a" if r.get(m) is None else f"{r[m]:g}") for m in metrics]
                 + [(lambda v: "n/a" if v is None else f"${v:.4f}")(_cost_per_query(r))] for r in latest_rows}
        st.dataframe(pd.DataFrame(table, index=metrics + ["cost / query"]), width="stretch")
        st.caption(f"Run {results.run_label(last)}")
with right:
    st.markdown("#### Hybrid vs keyword-only retrieval")
    h, b = results.latest(rows, model, "mistral"), results.latest(rows, model, "none")
    if h and b:
        m1, m2 = st.columns(2)
        m1.metric("Hit@6, hybrid", f"{h['hit@k %']:g}%", f"{h['hit@k %'] - b['hit@k %']:+.1f} pts vs BM25")
        m2.metric("False refusals, hybrid", f"{h['false refusals %']:g}%",
                  f"{h['false refusals %'] - b['false refusals %']:+.1f} pts vs BM25", delta_color="inverse")
        st.caption(f"{model}. With 32 in-scope questions, one question moves these by 3.1 points, "
                   "so read them as a direction rather than a measured effect.")
    else:
        st.caption("Run the eval with `--embed none` as well to see the ablation.")

st.markdown('<div class="rc-note"><b>Read with care.</b> n = 32 in-scope questions, so small differences are '
            "noise. The judge is mistral-large, which may favour its own answers. Every retrieval hit ranks first "
            "(MRR equals hit rate), which suggests the questions are worded close to the Article text and don't yet "
            "test ranking.</div>", unsafe_allow_html=True)

# ---------- every question ----------
st.markdown("#### Every question")
detail = results.load_rows(config.RESULTS_DIR, cur["stamp"], cur["backend"], model)
if detail is None:
    st.info("Per-question results for this run aren't in this checkout (results/*.jsonl). "
            "Run the eval to generate them.")
    st.stop()

golden = {g["id"]: g for g in map(json.loads, (config.ROOT / "eval" / "golden.jsonl").read_text().splitlines())}


def _outcome(r: dict) -> tuple[bool, str]:
    if r["out_of_scope"]:
        return (True, "declined out-of-scope") if r["refused"] else (False, "answered out-of-scope")
    if r["refused"]:
        return False, "false refusal"
    if not r["hit"]:
        return False, "retrieval miss"
    if not r["cited_expected"]:
        return False, "wrong citation"
    if r["ungrounded"]:
        return False, "cited unretrieved Article"
    if (r.get("judge") or 5) <= 3:
        return False, f"judge {r['judge']}/5"
    return True, ""


table = pd.DataFrame([{"ok": _outcome(r)[0], "result": "✓ pass" if _outcome(r)[0] else f"✗ {_outcome(r)[1]}",
                       "id": r["id"], "topic": r.get("topic", ""),
                       "question": r["question"], "judge": r.get("judge"), "latency s": r["latency_s"]}
                      for r in detail])
failures_only = st.toggle(f"Only show problems ({int((~table['ok']).sum())})", key="dd-fail")
view = table[~table["ok"]] if failures_only else table
event = st.dataframe(
    view.drop(columns="ok"), hide_index=True, width="stretch", on_select="rerun", selection_mode="single-row",
    key=f"dd-{cur['stamp']}-{model}",
    column_config={"result": st.column_config.TextColumn("Result", width="medium"),
                   "question": st.column_config.TextColumn("Question", width="large"),
                   "judge": st.column_config.NumberColumn("Judge", format="%d"),
                   "latency s": st.column_config.NumberColumn("Latency", format="%.1fs")})
picked = event.selection.rows if event and event.selection else []
if picked:
    r = next(d for d in detail if d["id"] == view.iloc[picked[0]]["id"])
    known = c.get_index("none").unit_ids if config.CHUNKS_PATH.exists() else None
    by_unit = c.chunks_by_unit("none") if known else {}
    with st.container(border=True):
        st.markdown(f"**{r['id']}** · {c.plain_md(r['question'])}")
        st.markdown(render.citation_html(r["answer"], set(r["retrieved"]), known, by_unit), unsafe_allow_html=True)
        expected = golden.get(r["id"], {}).get("expected", [])
        st.markdown(
            "**Expected** " + (", ".join(render.unit_label(u) for u in expected) or "a refusal")
            + " · **Retrieved** " + ", ".join(("**" + render.unit_label(u) + "**") if u in expected
                                              else render.unit_label(u) for u in r["retrieved"]))
        if r.get("judge") is not None:
            st.caption(f"Judge {r['judge']}/5: {r.get('judge_reason', '')}")
else:
    st.caption("Select a row to read the answer, with its citations checked.")
