# Handover: EU AI Act & DORA Copilot

This document gives a coding assistant everything it needs to keep working on this repository. Read it fully before changing any code.

## 1. What this is

A retrieval-augmented generation (RAG) assistant that answers questions about two EU regulations:

- **AI Act**: Regulation (EU) 2024/1689
- **DORA**: Regulation (EU) 2022/2554 (Digital Operational Resilience Act)

Every answer must cite the exact Article or Annex it relies on, for example `[AI Act Art. 27(1)]` or `[DORA Art. 28(3)]`. It runs on Mistral models in two modes:

- **Hosted:** Mistral La Plateforme API (`mistral-small-latest`, `mistral-large-latest`, `mistral-embed`)
- **Sovereign:** open-weight Mistral models run locally through Ollama, with nothing leaving the machine

It includes an evaluation harness that scores retrieval, citation accuracy, groundedness and refusals across models and modes.

## 2. Why it exists

It's a portfolio project for forward-deployed engineering and AI deployment roles. It needs to show, and stand up to discussion in a technical interview:

1. **RAG done properly for a regulated domain.** Chunking follows the structure of the law, retrieval is hybrid, and citations are verified.
2. **Evaluation discipline.** A golden question set and measurable metrics, so model choices rest on evidence.
3. **Deployment fit.** The same pipeline runs hosted or fully local, so the choice follows the customer's data, security and latency constraints.
4. **Value framing.** Comparing small and large models on cost, latency and quality, including when a smaller model is good enough (cheaper and lower-carbon).
5. **Customer-ready delivery.** Clear docs, tests that run offline, and a handover another team can pick up.

The target user is a compliance, risk or architecture team at a European financial institution. The code should stay easy to read and explain: clarity matters more than cleverness.

## 3. Current status

| Area | Status |
|---|---|
| Offline tests (`python tests/test_pipeline.py`) | ✅ 15 pass, 1 skipped (the UI render test needs Streamlit's AppTest ≥ 1.55). CI runs them on Linux and Windows, Python 3.9 and 3.12. |
| Ingestion of real EUR-Lex text | ✅ AI Act: 113 articles and 13 annexes (226 chunks). DORA: 64 articles (125 chunks). 351 chunks in total. |
| Golden set check (`eval/verify_golden.py`) | ✅ 0 problems, after changing the verify phrase for DORA Art. 26 from "threat-led penetration testing" to "TLPT" |
| Embedding index (`--embed mistral`) | ✅ Built locally |
| CLI `ask` and Streamlit app | ✅ Working end to end; the app has four pages (Ask, Compare, Evaluation, How it works) |
| Full evaluation run | ✅ Run 9–11 October 2026; results and failure analysis in the README |
| Sovereign mode (Ollama) | ✅ Measured with `mistral:7b` on an 8 GB M1 Mac mini (`OLLAMA_NUM_CTX=4096`); see the README |

**Environment:** macOS, VS Code, Python **3.9** venv (system Python). The code must keep working on 3.9; every module uses `from __future__ import annotations`. The `urllib3` `NotOpenSSLWarning` is harmless.

**Fetching:** EUR-Lex blocks scripts with a bot check. `regcopilot/fetch.py` (added after the first handover) gets the texts from the EU Publications Office's Cellar repository instead, falling back to content negotiation, then EUR-Lex, then any browser-saved file in `data/raw/`. Offline tests cover all the fallbacks. The live fetch succeeded on 2026-10-08 via `cellar-negotiation`. If every route fails, `ingest` falls back to the committed corpus snapshot in `corpus/` (`ingest --snapshot` uses it deliberately).

## 4. Repository map

```
eu-reg-copilot/
├── regcopilot/
│   ├── config.py      # paths, model names, .env loader (no python-dotenv dependency)
│   ├── sources.py     # the two regulations: CELEX ids, EUR-Lex URLs, short names
│   ├── fetch.py       # official texts from Cellar / EUR-Lex / saved file; validates by parsing; manifest.json
│   ├── ingest.py      # fetch, parse, chunk, write data/chunks.json
│   ├── parse.py       # EUR-Lex HTML -> Units (Article/Annex/Recital) -> Chunks
│   ├── bm25.py        # dependency-free BM25 with light stemming
│   ├── index.py       # hybrid retrieval: BM25 + embeddings, fused with RRF, plus explicit Article lookup
│   ├── llm.py         # MistralClient, OllamaClient and MockClient behind one interface (plain HTTP)
│   ├── rag.py         # system prompt, context building, answer(), citation parsing and grounding check
│   └── __main__.py    # CLI: ingest | index | ask
├── app.py             # Streamlit entry point with hosted/sovereign toggle
├── ui/                # app pages (views/: ask, compare, evaluation, how) and shared components
├── corpus/            # committed chunk snapshot + manifest, the ingest fallback
├── eval/
│   ├── golden.jsonl       # 32 in-scope and 3 out-of-scope questions, with expected units and reference answers
│   ├── verify_golden.py   # checks expected units exist and contain their verify phrase
│   └── run_eval.py        # metrics, multi-model comparison, optional LLM judge, markdown summary
├── tests/
│   ├── test_pipeline.py   # offline tests using the mock backend
│   └── fixtures/          # SYNTHETIC EUR-Lex-style HTML (paraphrased, not the legal text)
├── data/              # gitignored: raw/, chunks.json, index/<backend>/
├── results/           # eval outputs (*.jsonl gitignored, *_summary.md kept)
├── .github/workflows/ # CI: offline tests on Linux and Windows
├── .env               # gitignored: MISTRAL_API_KEY
├── .env.example, prices.example.json, requirements.txt, README.md
```

## 5. How the pipeline works

1. **Parse** (`parse.py`). `extract_lines` flattens the HTML into ordered lines; each table row becomes one line, because EUR-Lex uses tables for points such as "(a) …". `parse_units` is a line-based state machine driven by headings matching `^Article N$`, `^ANNEX [IVX]+$`, `^CHAPTER [IVX]+$` and `^SECTION N$`. It never relies on CSS class names. It skips the preamble (recitals are optional), the signature block after "Done at …", and footnotes after the signature.
2. **Chunk** (`chunk_units`). Chunks never cross an Article or Annex boundary. Long units split on numbered paragraphs (`^\d+\.`), with at most `MAX_CHUNK_WORDS` (350) words per chunk. Each chunk's text starts with a header such as `AI Act Art. 27 — <title> (Chapter III: …)`.
3. **Identifiers.** `unit_id` takes the form `AIA-Art-27`, `AIA-Annex-III` or `DORA-Art-28`. `label` takes the form `AI Act Art. 27`, `AI Act Annex III` or `DORA Art. 28`. Labels are the citation format the model must reproduce.
4. **Retrieve** (`index.py`). BM25 and cosine similarity each rank a pool of 50; the two rankings are combined with Reciprocal Rank Fusion (k=60). If the query names a unit ("DORA Article 28", "Annex III of the AI Act"), that unit is boosted to the top. Indexes are stored per embedding backend in `data/index/<backend>/`. `none` means BM25 only.
5. **Generate** (`rag.py`). Retrieved chunks are grouped by unit into `[SOURCE: <label>] (<title>)` blocks. The system prompt requires: use only the sources, cite every claim, reply with exactly `REFUSAL` if the answer isn't there, at most 200 words, British English.
6. **Verify.** `parse_citations` extracts the cited unit ids. Any cited unit that wasn't retrieved is listed in `ungrounded_citations` and shown in the UI and the evaluation.

## 6. Commands

```bash
source .venv/bin/activate
python tests/test_pipeline.py
python -m regcopilot fetch --refresh -v           # download the texts, showing each route
python -m regcopilot ingest                       # fetches if needed, then chunks
python eval/verify_golden.py
python -m regcopilot index --embed mistral        # or: ollama | none | mock
python -m regcopilot ask "..." --backend mistral  # or: ollama | mock
streamlit run app.py
python eval/run_eval.py --backend mistral --models mistral-small-latest,mistral-large-latest --judge mistral-large-latest
```

## 7. Rules for changes

- **Keep the offline tests passing** and add a test for any new behaviour. The mock backend must always work without network access or an API key.
- **Never print, log or commit `MISTRAL_API_KEY`.** `.env` and `data/` stay gitignored.
- **Citation format is a contract.** If you change `Unit.label`, also update `RE_CITE` in `rag.py`, the system prompt, `golden.jsonl` and the tests.
- **If you change parsing or chunking,** re-run `ingest` and `verify_golden.py`, and confirm the counts stay at 113 articles and 13 annexes (AI Act) and 64 articles (DORA).
- **Keep dependencies minimal:** `requests`, `beautifulsoup4`, `lxml`, `numpy`, `streamlit`. Explain any new dependency.
- **Keep Python 3.9 compatible.** No `match` statements; keep `from __future__ import annotations`.
- **Don't edit `eval/golden.jsonl` to make a score look better.** Change it only to fix a factual error, and note why in the commit message.
- **Never put Mistral pricing in code.** Prices belong in `prices.json`, filled in by the user from mistral.ai/pricing.
- Write docs and UI text in British English. Answers are not legal advice, and the UI says so.

## 8. Suggested next tasks (in priority order)

1. **Confirm automatic fetching live.** Run `python -m regcopilot fetch --refresh -v` and record which strategy succeeded for each regulation. If the `cellar-item` URL fails, inspect Cellar's response (for example a 300 list or a zip) and adjust `fetch.py`, adding a test.
2. **Run the full evaluation** for small vs large (hosted) and keyword-only vs hybrid retrieval, then paste the results into the README table. Look at failures in `results/*.jsonl` and categorise them as retrieval misses, wrong citations or false refusals.
3. **Sovereign mode check.** Install Ollama, pull `mistral-nemo` and `nomic-embed-text`, build the `ollama` index, run the evaluation, and add the row to the README.
4. **CI.** Add a GitHub Actions workflow that runs `python tests/test_pipeline.py` on Python 3.9 and 3.12.
5. **Re-ranking.** Optionally re-score the top 20 candidates with a small Mistral model before generation, behind a flag, and measure whether hit@k and the judge score improve.
6. **Multilingual.** Ingest the French text (`/FR/` in the EUR-Lex URL), allow questions in French or English, and cite consistently. Mistral's multilingual strength is part of the story.
7. **Second-tier sources.** Add DORA technical standards (RTS/ITS) and ESA guidance as a separate, lower-priority corpus with its own labels.
8. **Packaging.** Add a Dockerfile for an air-gapped, sovereign deployment that bundles the app with Ollama.

## 9. Things to be careful about

- The reference answers in `golden.jsonl` were written by hand. Treat any disagreement with the legal text as a bug in the golden set and fix it, as was done for DORA Art. 26.
- The corpus is the original published text. Later amendments or changes to application dates aren't reflected, and the README's limitations section says so.
- EUR-Lex HTML has repeated headings and footnote blocks. The parser de-duplicates units and keeps the longest version; don't remove that step.
