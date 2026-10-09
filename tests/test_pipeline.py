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
    units = {u.unit_id: u for u in parse_units((FIX / "AIA_sample.html").read_text(), "AIA", include_recitals=True)}
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
    units = parse_units((FIX / "AIA_sample.html").read_text(), "AIA")
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
        idx = Index.build(__import__("json").loads(config.CHUNKS_PATH.read_text()), embed)
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
    a = answer("Who is responsible for ICT risk?", Index.load("mock"), MockClient(), k=3)
    assert a.citations and not a.ungrounded_citations and not a.refused


def test_citation_parsing():
    from regcopilot.rag import is_refusal, parse_citations
    text = "Deployers must assess impact [AI Act Art. 27(1)] and see [AI Act Annex III, point 5(c); DORA Art. 28(3)]."
    assert parse_citations(text) == ["AIA-Art-27", "AIA-Annex-III", "DORA-Art-28"]
    assert parse_citations("no citations here") == []
    assert is_refusal("I can’t find this in the AI Act or DORA texts I have.")


def test_eval_metrics_on_mock():
    _setup_tmp()
    import json
    from regcopilot.index import Index
    from regcopilot.ingest import ingest
    sys.path.insert(0, str(ROOT / "eval"))
    from run_eval import evaluate, to_markdown
    from verify_golden import load_golden, verify
    ingest(html_paths={"AIA": FIX / "AIA_sample.html", "DORA": FIX / "DORA_sample.html"})
    chunks = json.loads(config.CHUNKS_PATH.read_text())
    golden = [g for g in load_golden() if g["id"] in {"aia-03", "aia-07", "aia-16", "dora-05", "dora-07", "dora-09"}]
    assert not verify(golden, chunks), verify(golden, chunks)
    idx = Index.build(chunks, "mock")
    rows, summary = evaluate("mock", "mock-extractive", idx, golden, k=4)
    assert summary["hit@k %"] == 100.0, summary
    assert summary["grounded %"] == 100.0
    assert "| model |" in to_markdown([summary])


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
    manifest = json.loads((config.RAW_DIR / "manifest.json").read_text())
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
    for t in tests:
        t()
        print("PASS", t.__name__)
    print(f"{len(tests)} tests passed")
