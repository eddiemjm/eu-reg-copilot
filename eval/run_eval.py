"""Evaluate retrieval and answers across models and backends.

    # compare two hosted Mistral models, with an LLM judge
    python eval/run_eval.py --backend mistral --models mistral-small-latest,mistral-large-latest --judge mistral-large-latest

    # sovereign mode: local open-weight model via Ollama, judged by the hosted API
    python eval/run_eval.py --backend ollama --models mistral-nemo --embed ollama --judge mistral-large-latest

Metrics
  retrieval hit@k   an expected Article/Annex is among the retrieved sources
  MRR               mean reciprocal rank of the first expected source
  cited correctly   the answer cites at least one expected Article/Annex
  grounded          every citation points to a source that was actually retrieved (no invented citations)
  nonexistent cites citations to Articles/Annexes that don't exist in the regulation at all
  false refusals    in-scope questions the model wrongly declined
  OOS refusals      out-of-scope questions correctly declined
  judge (1-5)       optional LLM-as-judge score against the hand-written reference answer
  latency, tokens, cost (if prices.json is filled in)
"""
from __future__ import annotations

import argparse
import json
import statistics
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from regcopilot import config  # noqa: E402
from regcopilot.index import Index  # noqa: E402
from regcopilot.llm import get_client  # noqa: E402
from regcopilot.rag import finalise, prepare  # noqa: E402
from verify_golden import load_golden  # noqa: E402

JUDGE_PROMPT = """You are grading an answer about EU regulation against a reference answer written by an expert.
Score 1-5:
5 = correct and complete on the key points, no errors
4 = correct, minor omissions
3 = partly correct or missing important points
2 = mostly incorrect or misleading
1 = wrong, or refuses when an answer was possible
Return JSON only: {{"score": <1-5>, "reason": "<one sentence>"}}

Question: {q}
Reference answer: {ref}
Answer to grade: {ans}"""


def judge(client, model: str, q: str, ref: str, ans: str) -> tuple[int | None, str]:
    try:
        r = client.chat([{"role": "user", "content": JUDGE_PROMPT.format(q=q, ref=ref, ans=ans)}],
                        model=model, json_mode=True, max_tokens=200)
        data = json.loads(r.text)
        return int(data["score"]), str(data.get("reason", ""))
    except Exception as e:  # a judge failure should not kill the run
        return None, f"judge error: {e}"


def evaluate(backend: str, model: str, index: Index, golden: list[dict], k: int,
             judge_client=None, judge_model: str | None = None) -> tuple[list[dict], dict]:
    client = get_client(backend)
    rows = []
    for g in golden:
        oos = bool(g.get("out_of_scope"))
        expected = set(g.get("expected", []))
        # A failed question must not kill the run: retrieval metrics are computed whenever
        # retrieval succeeded, and generation failures become error rows.
        try:
            hits, messages = prepare(g["question"], index, k=k)
        except Exception as e:
            rows.append({"id": g["id"], "topic": g.get("topic", ""), "question": g["question"],
                         "out_of_scope": oos, "answer": "", "refused": None, "citations": [],
                         "ungrounded": [], "nonexistent": [], "retrieved": [], "hit": None,
                         "rr": None, "cited_expected": None, "prompt_tokens": 0,
                         "completion_tokens": 0, "latency_s": 0.0, "error": str(e).splitlines()[0][:200]})
            print(f"  XX {g['id']:8} retrieval error: {str(e).splitlines()[0][:80]}")
            continue
        retrieved = list(dict.fromkeys(h.chunk["unit_id"] for h in hits))
        ranks = [i for i, u in enumerate(retrieved) if u in expected]
        row = {
            "id": g["id"], "topic": g.get("topic", ""), "question": g["question"], "out_of_scope": oos,
            "retrieved": retrieved,
            "hit": bool(ranks) if not oos else None,
            "rr": (1.0 / (ranks[0] + 1) if ranks else 0.0) if not oos else None,
        }
        try:
            res = client.chat(messages, model=model)
            a = finalise(g["question"], res.text, hits, res, known_units=index.unit_ids)
            row.update({
                "answer": a.text, "refused": a.refused, "citations": a.citations,
                "ungrounded": a.ungrounded_citations, "nonexistent": a.nonexistent_citations,
                "cited_expected": bool(expected & set(a.citations)) if not oos else None,
                "prompt_tokens": a.prompt_tokens, "completion_tokens": a.completion_tokens,
                "latency_s": round(a.latency_s, 3),
            })
            if judge_client is not None and not oos:
                row["judge"], row["judge_reason"] = judge(judge_client, judge_model, g["question"],
                                                          g["reference"], a.text)
            flag = "OK " if (row["refused"] if oos else row["cited_expected"]) else "-- "
            print(f"  {flag}{g['id']:8} {a.latency_s:5.1f}s  cites={','.join(a.citations) or '-'}")
        except Exception as e:
            row.update({
                "answer": "", "refused": None, "citations": [], "ungrounded": [],
                "nonexistent": [], "cited_expected": None, "prompt_tokens": 0,
                "completion_tokens": 0, "latency_s": 0.0, "error": str(e).splitlines()[0][:200],
            })
            print(f"  XX {g['id']:8} error: {str(e).splitlines()[0][:80]}")
        rows.append(row)
    return rows, summarise(rows, model)


def summarise(rows: list[dict], model: str) -> dict:
    ins = [r for r in rows if not r["out_of_scope"]]
    oos = [r for r in rows if r["out_of_scope"]]
    pct = lambda xs: round(100 * sum(1 for x in xs if x) / len(xs), 1) if xs else None  # noqa: E731
    non_null = lambda key, rows_: [r[key] for r in rows_ if r.get(key) is not None]  # noqa: E731
    judged = [r["judge"] for r in ins if r.get("judge") is not None]
    prices = config.load_prices().get(model)
    tin = sum(r["prompt_tokens"] for r in rows)
    tout = sum(r["completion_tokens"] for r in rows)
    cost = None
    if prices:
        cost = round(tin / 1e6 * prices["input_per_m"] + tout / 1e6 * prices["output_per_m"], 4)
    return {
        "model": model, "n": len(rows),
        "hit@k %": pct(non_null("hit", ins)),
        "MRR": round(statistics.mean(non_null("rr", ins)), 3) if any(r.get("rr") is not None for r in ins) else None,
        "cited correctly %": pct(non_null("cited_expected", ins)),
        "grounded %": pct([not r["ungrounded"] for r in rows if r["citations"]]),
        "nonexistent cites": sum(len(r.get("nonexistent", [])) for r in rows),
        "false refusals %": pct(non_null("refused", ins)),
        "OOS refused %": pct(non_null("refused", oos)),
        "judge (1-5)": round(statistics.mean(judged), 2) if judged else None,
        "p50 latency s": round(statistics.median(r["latency_s"] for r in rows), 2) if rows else None,
        "tokens in/out": f"{tin}/{tout}",
        "cost": cost,
    }


def to_markdown(summaries: list[dict]) -> str:
    cols = list(summaries[0])
    out = ["| " + " | ".join(cols) + " |", "|" + "---|" * len(cols)]
    for s in summaries:
        out.append("| " + " | ".join("n/a" if s[c] is None else str(s[c]) for c in cols) + " |")
    return "\n".join(out)


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--backend", default="mistral", choices=["mistral", "ollama", "mock"])
    p.add_argument("--models", default=None, help="comma-separated; defaults to the backend's default model")
    p.add_argument("--embed", default=None, help="index: mistral | ollama | mock | none (default: backend's if built)")
    p.add_argument("-k", type=int, default=config.TOP_K)
    p.add_argument("--judge", default=None, help="judge model (uses the Mistral API unless --judge-backend)")
    p.add_argument("--judge-backend", default="mistral")
    p.add_argument("--limit", type=int, default=None)
    p.add_argument("--golden", default=str(Path(__file__).with_name("golden.jsonl")))
    args = p.parse_args()

    golden = load_golden(Path(args.golden))[: args.limit]
    embed = args.embed or (args.backend if args.backend in Index.available() else "none")
    index = Index.load(embed)
    default_model = get_client(args.backend).default_chat_model
    models = [m.strip() for m in (args.models or default_model).split(",")]
    judge_client = get_client(args.judge_backend) if args.judge else None

    config.RESULTS_DIR.mkdir(exist_ok=True)
    stamp = time.strftime("%Y%m%d-%H%M%S")
    summaries = []
    for m in models:
        print(f"\n== {args.backend}:{m}  (index: {embed}, k={args.k}, {len(golden)} questions)")
        rows, summary = evaluate(args.backend, m, index, golden, args.k, judge_client, args.judge)
        summary["backend"], summary["index"] = args.backend, embed
        summaries.append(summary)
        out = config.RESULTS_DIR / f"{stamp}_{args.backend}_{m.replace('/', '-').replace(':', '-')}.jsonl"
        out.write_text("\n".join(json.dumps(r, ensure_ascii=False) for r in rows), encoding="utf-8")
        print(f"  details -> {out}")

    md = to_markdown(summaries)
    (config.RESULTS_DIR / f"{stamp}_summary.md").write_text(md + "\n", encoding="utf-8")
    print("\n" + md)
    return 0


if __name__ == "__main__":
    sys.exit(main())
