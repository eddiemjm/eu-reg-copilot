"""Offline tests: no network, no API key. Run with `python tests/test_pipeline.py` or `pytest`."""
from __future__ import annotations

import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
FIX = Path(__file__).with_name("fixtures")

from regcopilot import config  # noqa: E402


def _setup_tmp():
    tmp = Path(tempfile.mkdtemp())
    config.DATA_DIR = tmp
    config.CHUNKS_PATH = tmp / "chunks.json"
    config.RAW_DIR = tmp / "raw"
    config.INDEX_DIR = tmp / "index"
    import regcopilot.index as ix
    ix.config = config
    return tmp


def test_parse_structure():
    from regcopilot.parse import parse_units
    units = {u.unit_id: u for u in parse_units((FIX / "AIA_sample.html").read_text(encoding="utf-8"), "AIA", include_recitals=True)}
    assert {"AIA-Art-3", "AIA-Art-4", "AIA-Art-5", "AIA-Art-27", "AIA-Art-99", "AIA-Annex-III"} <= set(units)
    assert "AIA-Rec-1" in units and "AIA-Rec-2" in units
    assert units["AIA-Art-5"].title == "Prohibited AI practices"
    assert units["AIA-Art-5"].chapter == "II" and units["AIA-Art-5"].chapter_title == "PROHIBITED AI PRACTICES"
    assert any("(f) AI systems to infer emotions" in l for l in units["AIA-Art-5"].lines)  # table rows -> lines
    annex = " ".join(units["AIA-Annex-III"].lines)
    assert "life and health insurance" in annex
    assert not any("SECTION" in l or "Obligations of providers and deployers" in l for u in units.values() for l in u.lines)
    assert units["AIA-Art-27"].chapter == "III"
    assert "OJ C 999" not in annex and "The President" not in annex  # footnotes and signatures dropped


def test_chunking_respects_units_and_paragraphs():
    from regcopilot.parse import chunk_units, parse_units
    units = parse_units((FIX / "AIA_sample.html").read_text(encoding="utf-8"), "AIA")
    chunks = chunk_units(units, max_words=120)
    art27 = [c for c in chunks if c.unit_id == "AIA-Art-27"]
    assert len(art27) >= 2, "long Article should be split"
    assert all(c.text.startswith("AI Act Art. 27 — Fundamental rights impact assessment") for c in art27)
    assert art27[0].paragraphs[0] == "1"
    assert all(c.unit_id.startswith("AIA-") for c in chunks)


def test_ingest_index_search_and_answer():
    _setup_tmp()
    from regcopilot.index import Index
    from regcopilot.ingest import ingest
    from regcopilot.llm import MockClient
    from regcopilot.rag import answer
    ingest(html_paths={"AIA": FIX / "AIA_sample.html", "DORA": FIX / "DORA_sample.html"})
    for embed in ("none", "mock"):
        idx = Index.build(__import__("json").loads(config.CHUNKS_PATH.read_text(encoding="utf-8")), embed)
        assert Index.load(embed).chunks
        top = [h.chunk["unit_id"] for h in idx.search("How often is threat-led penetration testing required?", k=3)]
        assert top[0] == "DORA-Art-26", top
        top = [h.chunk["unit_id"] for h in idx.search("emotion recognition at work", k=3)]
        assert "AIA-Art-5" in top, top
        # explicit reference boost
        top = [h.chunk["unit_id"] for h in idx.search("What does DORA Article 28 say?", k=2)]
        assert top[0] == "DORA-Art-28", top
        top = [h.chunk["unit_id"] for h in idx.search("Summarise Annex III of the AI Act", k=2)]
        assert top[0] == "AIA-Annex-III", top
        # a long named Article can't take every slot
        hits = idx.search("What does AI Act Article 27 say about insurers?", k=3)
        units = [h.chunk["unit_id"] for h in hits]
        assert units[0] == "AIA-Art-27" and units.count("AIA-Art-27") == 1, units
    a = answer("Who is responsible for ICT risk?", Index.load("mock"), MockClient(), k=3)
    assert a.citations and not a.ungrounded_citations and not a.refused and not a.nonexistent_citations


def test_nonexistent_citations_are_separated():
    from regcopilot.llm import ChatResult
    from regcopilot.rag import finalise
    res = ChatResult("", "m", 0, 0, 0.0)
    a = finalise("q", "See [DORA Art. 99] and [DORA Art. 28].", [], res, known_units={"DORA-Art-28"})
    assert a.ungrounded_citations == ["DORA-Art-99", "DORA-Art-28"]
    assert a.nonexistent_citations == ["DORA-Art-99"]


def test_citation_parsing():
    from regcopilot.rag import is_refusal, parse_citations
    text = "Deployers must assess impact [AI Act Art. 27(1)] and see [AI Act Annex III, point 5(c); DORA Art. 28(3)]."
    assert parse_citations(text) == ["AIA-Art-27", "AIA-Annex-III", "DORA-Art-28"]
    assert parse_citations("no citations here") == []
    # parenthesised citations count too (they used to slip past both scoring and grounding)
    assert parse_citations("Contracts must set SLAs (DORA Art. 30(2)) and exit plans (DORA Article 28(8)).") \
        == ["DORA-Art-30", "DORA-Art-28"]
    assert parse_citations("The DORA article in question and the AI Act annex cover this.") == []
    # follow-on citations inherit the regulation, so they're checked too
    assert parse_citations("AI Act Art. 16(a), Art. 26(5)") == ["AIA-Art-16", "AIA-Art-26"]
    assert parse_citations("AI Act Articles 9, 10 and 15(1)") == ["AIA-Art-9", "AIA-Art-10", "AIA-Art-15"]
    assert parse_citations("AI Act Art. 5 and DORA Art. 28") == ["AIA-Art-5", "DORA-Art-28"]
    assert parse_citations("AI Act Art. 26(5), (6), (7)") == ["AIA-Art-26"]
    assert parse_citations("under DORA Art. 19, 30 days later") == ["DORA-Art-19"]
    assert is_refusal("I can’t find this in the AI Act or DORA texts I have.")
    # paraphrased refusal scores as a refusal
    assert is_refusal("I can't find the Solvency II calibration in the provided AI Act or DORA texts. "
                      "The sources provided pertain to digital operational resilience.")
    # declining and then answering from memory is a failure, not a refusal
    assert not is_refusal("I can't find this in the AI Act or DORA texts I have. The maximum fine under "
                          "the GDPR is 4% of global annual turnover or EUR 20 million.")
    # describing what the sources don't contain, without figures, is still a refusal
    assert is_refusal("I can't find this in the AI Act or DORA texts I have. The sources provided do "
                      "not contain information about the 2022 FIFA World Cup.")
    # a partly answered reply that cites something is an answer, not a refusal
    assert not is_refusal("I can't find a specific deadline in the sources, but providers must register "
                          "the system [AI Act Art. 49(1)].")
    # additions from memory leak even without figures: obligations or dates after the decline
    assert not is_refusal("I can't find this in the AI Act or DORA texts I have. Under the GDPR, a "
                          "controller must notify the supervisory authority without undue delay.")
    assert not is_refusal("I can't find this in the AI Act or DORA texts I have. The obligation applies "
                          "from 2 August 2026.")


def test_eval_metrics_on_mock():
    _setup_tmp()
    import json
    from regcopilot.index import Index
    from regcopilot.ingest import ingest
    sys.path.insert(0, str(ROOT / "eval"))
    from run_eval import evaluate, to_markdown
    from verify_golden import load_golden, verify
    ingest(html_paths={"AIA": FIX / "AIA_sample.html", "DORA": FIX / "DORA_sample.html"})
    chunks = json.loads(config.CHUNKS_PATH.read_text(encoding="utf-8"))
    golden = [g for g in load_golden() if g["id"] in {"aia-03", "aia-07", "aia-16", "dora-05", "dora-07", "dora-09"}]
    assert not verify(golden, chunks), verify(golden, chunks)
    idx = Index.build(chunks, "mock")
    rows, summary = evaluate("mock", "mock-extractive", idx, golden, k=4)
    assert summary["hit@k %"] == 100.0, summary
    assert summary["grounded %"] == 100.0
    assert "| model |" in to_markdown([summary])


def test_citation_pills():
    from ui.render import citation_html
    text = ("Assess impact [AI Act Art. 27(1); DORA Art. 28(3)], set SLAs (DORA Art. 30(2)), see [DORA Art. 99]. "
            "<script>alert(1)</script> costs $5")
    out = citation_html(text, retrieved={"AIA-Art-27", "DORA-Art-30"},
                        known={"AIA-Art-27", "DORA-Art-28", "DORA-Art-30"})
    assert '<span class="rc-pill ok"' in out and "✓ AI Act Art. 27(1)</span>" in out
    assert '<span class="rc-pill missing"' in out and "! DORA Art. 28(3)</span>" in out
    assert "✓ DORA Art. 30(2)</span>)" in out                     # parenthesised citation, suffix kept
    assert '<span class="rc-pill invented"' in out                # DORA has 64 Articles
    assert "<script>" not in out and "&lt;script&gt;" in out and "\\$5" in out
    assert "\n" not in out                                        # a blank line would let Markdown split the tags
    ranged = citation_html("logging [AI Act Art. 26(5)-(12)].", retrieved={"AIA-Art-26"})
    assert "✓ AI Act Art. 26(5)-(12)</span>." in ranged and "[" not in ranged  # whole range in one pill


def test_summary_parsing_old_and_new_columns():
    from ui import results
    old = "| model | n | hit@k % | grounded % | cost | backend | index |\n|---|---|---|---|---|---|---|\n" \
          "| mistral-small-latest | 35 | 90.6 | 100.0 | n/a | mistral | none |"
    new = "| model | n | hit@k % | grounded % | nonexistent cites | cost | backend | index |\n|---|---|---|---|---|---|---|---|\n" \
          "| mistral-small-latest | 35 | 93.8 | 100.0 | 0 | 0.0147 | mistral | mistral |"
    d = Path(tempfile.mkdtemp())
    (d / "20261009-131917_summary.md").write_text(old, encoding="utf-8")
    (d / "20261011-185938_summary.md").write_text(new, encoding="utf-8")
    (d / "notes_summary.md").write_text(new, encoding="utf-8")                      # not a run: ignored
    rows = results.load_summaries(d)
    assert [r["stamp"] for r in rows] == ["20261009-131917", "20261011-185938"]
    assert rows[0]["cost"] is None and "nonexistent cites" not in rows[0]
    assert rows[1]["nonexistent cites"] == 0 and rows[1]["index"] == "mistral"
    assert results.latest(rows, "mistral-small-latest", "mistral")["hit@k %"] == 93.8
    assert results.latest(rows, "mistral-small-latest", "none", before="20261009-131917") is None


def _app():
    from streamlit.testing.v1 import AppTest
    return AppTest.from_file(str(ROOT / "app.py"), default_timeout=60)


def test_ui_pages_render_offline():
    import json
    import os
    import streamlit
    if tuple(int(x) for x in streamlit.__version__.split(".")[:2]) < (1, 55):
        # AppTest before 1.55 can't rerun an app with a multi-character st.segmented_control option.
        return f"needs Streamlit's AppTest >= 1.55 (have {streamlit.__version__})"
    os.environ["REGCOPILOT_UI_MOCK"] = "1"
    _setup_tmp()
    from regcopilot.index import Index
    from regcopilot.ingest import ingest
    ingest(html_paths={"AIA": FIX / "AIA_sample.html", "DORA": FIX / "DORA_sample.html"})
    Index.build(json.loads(config.CHUNKS_PATH.read_text(encoding="utf-8")), "mock")
    at = _app()
    at.run()
    for page in ("ui/views/compare.py", "ui/views/evaluation.py", "ui/views/how.py"):
        assert not at.exception, [e.message for e in at.exception]
        at.switch_page(page).run()
    assert not at.exception, [e.message for e in at.exception]

    # Ask a question end to end with the mock model: pills, badge and evidence all render.
    at = _app()
    at.session_state["mode"] = "Offline mock"
    at.run()
    at.button(key="ex-1").click().run()
    assert not at.exception, [e.message for e in at.exception]
    history = at.session_state["history"]
    assert len(history) == 1 and history[0].citations
    assert any("rc-pill" in m.value for m in at.markdown)


def test_ui_without_corpus_shows_setup():
    _setup_tmp()                                                  # fresh empty data dir
    at = _app()
    at.run()
    assert not at.exception, [e.message for e in at.exception]
    assert any("corpus hasn't been built" in w.value for w in at.warning)


def test_ingest_falls_back_to_committed_snapshot():
    import json
    import regcopilot.ingest as ig
    from regcopilot.ingest import ingest
    _setup_tmp()                                                  # empty data dir: no cache, no saved file
    real_fetch = ig.fetch
    ig.fetch = lambda *a, **k: (_ for _ in ()).throw(RuntimeError("offline"))  # every route fails
    try:
        chunks = ingest()
    finally:
        ig.fetch = real_fetch
    assert len(chunks) >= 300 and {c["source"] for c in chunks} == {"AIA", "DORA"}
    manifest = json.loads((config.RAW_DIR / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["AIA"]["strategy"] == "committed-snapshot" and len(manifest["AIA"]["sha256"]) == 64
    assert manifest["DORA"]["strategy"] == "committed-snapshot" and "cellar" in manifest["DORA"]["url"]


def test_ingest_snapshot_mode():
    import json
    from regcopilot.ingest import ingest
    _setup_tmp()
    chunks = ingest(snapshot=True)                                # deliberate snapshot, no fetch
    assert len(chunks) >= 300 and {c["source"] for c in chunks} == {"AIA", "DORA"}
    manifest = json.loads((config.RAW_DIR / "manifest.json").read_text(encoding="utf-8"))
    assert all(manifest[s]["strategy"] == "committed-snapshot" for s in ("AIA", "DORA"))
    # the snapshot has one fixed chunk size: asking for a different one must refuse, not degrade
    try:
        ingest(snapshot=True, max_words=100)
        raise AssertionError("should have refused")
    except RuntimeError as e:
        assert "max_words" in str(e)


def test_eval_summary_with_error_rows():
    sys.path.insert(0, str(ROOT / "eval"))
    from run_eval import summarise
    row = {"id": "x", "out_of_scope": False, "hit": True, "rr": 1.0, "cited_expected": True,
           "citations": ["AIA-Art-1"], "ungrounded": [], "nonexistent": [], "refused": False,
           "prompt_tokens": 10, "completion_tokens": 5, "latency_s": 1.0}
    err = dict(row, id="y", hit=None, rr=None, cited_expected=None, refused=None, citations=[],
               error="context guard")
    s = summarise([row, err], "m")
    assert s["n"] == 2 and s["hit@k %"] == 100.0 and s["false refusals %"] == 0.0  # Nones excluded


def test_ollama_context_guard():
    import regcopilot.llm as llm_mod
    from regcopilot.llm import _ollama_options
    msgs = [{"role": "user", "content": "x" * 4000}]              # ~1,000 estimated tokens
    opts = _ollama_options(msgs, temperature=0.0, max_tokens=700)
    assert opts["num_ctx"] == config.OLLAMA_NUM_CTX and opts["num_predict"] == 700
    old = config.OLLAMA_NUM_CTX
    config.OLLAMA_NUM_CTX = 1000                                  # prompt + 700 max tokens > 1000
    try:
        _ollama_options(msgs, temperature=0.0, max_tokens=700)
        raise AssertionError("should have refused a prompt that would be truncated")
    except RuntimeError as e:
        assert "truncat" in str(e) and "OLLAMA_NUM_CTX" in str(e)
    finally:
        config.OLLAMA_NUM_CTX = old


def _resp(status=200, body=b"", ctype="text/html; charset=utf-8", url=""):
    import requests
    r = requests.Response()
    r.status_code, r._content, r.url = status, body, url
    r.headers["Content-Type"] = ctype
    r.encoding = "utf-8" if "charset=utf-8" in ctype else None
    return r


def test_fetch_falls_back_and_validates():
    import io
    import json
    import zipfile
    _setup_tmp()
    from regcopilot.fetch import fetch
    law = (FIX / "AIA_sample.html").read_bytes()
    bot_page = b"<html><body>Please verify you are human</body></html>"
    calls = []

    def fake_get(url, **kw):
        calls.append(url)
        if url.endswith(".fmx.xml.html") and "items/" not in url:
            return _resp(200, bot_page, url=url)                     # cellar-item: challenge page
        if url.endswith("/32024R1689"):                               # negotiation: 300 list
            return _resp(300, b'<a href="https://x.test/items/doc.zip">zip</a>', url=url)
        if url.endswith("doc.zip"):
            buf = io.BytesIO()
            with zipfile.ZipFile(buf, "w") as z:
                z.writestr("L_202401689EN.000101.fmx.xml.html", law)
            return _resp(200, buf.getvalue(), ctype="application/zip", url=url)
        return _resp(403, b"blocked", url=url)

    path = fetch("AIA", get=fake_get, min_articles=3)
    assert path.read_bytes() == law
    manifest = json.loads((config.RAW_DIR / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["AIA"]["strategy"] == "cellar-negotiation" and len(manifest["AIA"]["sha256"]) == 64
    # cached copy is reused without any network call
    calls.clear()
    fetch("AIA", get=fake_get, min_articles=3)
    assert calls == []


def test_fetch_uses_browser_saved_file_then_fails_cleanly():
    _setup_tmp()
    from regcopilot.fetch import fetch
    blocked = lambda url, **kw: _resp(403, b"blocked", url=url)  # noqa: E731
    try:
        fetch("DORA", get=blocked, min_articles=3)
        raise AssertionError("should have failed")
    except RuntimeError as e:
        assert "cellar-item" in str(e) and "eurlex-html" in str(e)
    config.RAW_DIR.mkdir(parents=True, exist_ok=True)
    saved = config.RAW_DIR / "L_2022333EN.01000101.xml.html"   # name the browser chooses
    saved.write_bytes((FIX / "DORA_sample.html").read_bytes())
    path = fetch("DORA", get=blocked, min_articles=3)
    assert path.name == "DORA.html" and path.read_bytes() == saved.read_bytes()


if __name__ == "__main__":
    tests = [v for k, v in dict(globals()).items() if k.startswith("test_")]
    skipped = 0
    for t in tests:
        reason = t()  # a test returns a reason string to mark itself skipped
        skipped += bool(reason)
        print(f"SKIP {t.__name__}: {reason}" if reason else f"PASS {t.__name__}")
    print(f"{len(tests) - skipped} tests passed" + (f", {skipped} skipped" if skipped else ""))
