"""Fetch the official regulation texts automatically, without a browser.

EUR-Lex's website protects itself with a bot check, so scripts often get a challenge
page instead of the law. The same official texts are published, for machines, by the EU
Publications Office's **Cellar** repository (publications.europa.eu), which EUR-Lex itself
is built on. So we try, in order:

1. cellar-item         the exact XHTML file for the English Official Journal text
2. cellar-negotiation  Cellar content negotiation: CELEX number + "give me English XHTML"
3. eurlex-html         the EUR-Lex HTML page (works when the bot check doesn't trigger)
4. saved-file          a page the user saved from a browser into data/raw/ (any filename)

Every candidate is *validated by parsing it*: it only counts if it yields roughly the
expected number of Articles. A bot-check page or an error page can never slip through.
Successful downloads are cached in data/raw/ and recorded in data/raw/manifest.json with
URL, strategy, SHA-256 and date, so the corpus is reproducible and auditable.
"""
from __future__ import annotations

import hashlib
import io
import json
import re
import shutil
import time
import zipfile
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import requests

from . import config
from .parse import parse_units
from .sources import SOURCES

CELLAR = "https://publications.europa.eu/resource/celex"
UA = "eu-reg-copilot/1.1 (research prototype; python-requests)"
BROWSER_UA = ("Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 "
              "(KHTML, like Gecko) Chrome/126.0 Safari/537.36")
XHTML_ACCEPT = "application/xhtml+xml, text/html;q=0.9, application/zip;q=0.5, */*;q=0.1"

Getter = Callable[..., requests.Response]


@dataclass
class Attempt:
    strategy: str
    url: str
    ok: bool
    detail: str


def _decode(content: bytes, declared: str | None) -> str:
    for enc in (declared, "utf-8", "latin-1"):
        if not enc:
            continue
        try:
            return content.decode(enc)
        except (UnicodeDecodeError, LookupError):
            continue
    return content.decode("utf-8", errors="replace")


def _from_zip(content: bytes) -> str | None:
    """Cellar returns a zip when a manifestation has several files; take the main (X)HTML one."""
    try:
        with zipfile.ZipFile(io.BytesIO(content)) as z:
            names = [n for n in z.namelist() if n.lower().endswith((".html", ".xhtml", ".htm"))]
            if not names:
                return None
            main = max(names, key=lambda n: z.getinfo(n).file_size)
            return _decode(z.read(main), "utf-8")
    except zipfile.BadZipFile:
        return None


def _links_from_300(text: str, base: str) -> list[str]:
    """A '300 Multiple Choices' response lists the available items; prefer (X)HTML ones."""
    links = re.findall(r'href="([^"]+)"', text) + re.findall(r'rdf:resource="([^"]+)"', text)
    links = [requests.compat.urljoin(base, l) for l in links]
    return sorted(set(links), key=lambda l: (not re.search(r"x?html?$", l, re.I), len(l)))


def _response_text(resp: requests.Response) -> str | None:
    ctype = resp.headers.get("Content-Type", "").lower()
    if "zip" in ctype or resp.content[:2] == b"PK":
        return _from_zip(resp.content)
    return _decode(resp.content, resp.encoding)


def validate(text: str | None, source: str, min_articles: int | None = None) -> tuple[bool, str]:
    if not text:
        return False, "empty response"
    need = min_articles if min_articles is not None else SOURCES[source]["min_articles"]
    try:
        units = parse_units(text, source)
    except Exception as e:  # malformed markup
        return False, f"unparseable ({type(e).__name__})"
    n = sum(1 for u in units if u.kind == "article")
    if n < need:
        hint = " (looks like a bot-check or error page)" if len(text) < 50_000 else ""
        return False, f"only {n} articles found, expected >= {need}{hint}"
    return True, f"{n} articles"


def _strategies(source: str) -> list[tuple[str, str, dict]]:
    s = SOURCES[source]
    out = []
    if s.get("cellar_item"):
        out.append(("cellar-item", f"{CELLAR}/{s['celex']}.ENG.xhtml.{s['cellar_item']}",
                    {"User-Agent": UA, "Accept": XHTML_ACCEPT}))
    out.append(("cellar-negotiation", f"{CELLAR}/{s['celex']}",
                {"User-Agent": UA, "Accept": XHTML_ACCEPT, "Accept-Language": "eng"}))
    out.append(("eurlex-html", s["html_url"],
                {"User-Agent": BROWSER_UA, "Accept": "text/html,application/xhtml+xml", "Accept-Language": "en"}))
    return out


def _try_url(strategy: str, url: str, headers: dict, source: str, get: Getter,
             min_articles: int | None) -> tuple[str | None, Attempt]:
    try:
        resp = get(url, headers=headers, timeout=120, allow_redirects=True)
    except requests.RequestException as e:
        return None, Attempt(strategy, url, False, type(e).__name__)
    if resp.status_code == 300:  # pick an (X)HTML item from the list and follow it
        for link in _links_from_300(_decode(resp.content, resp.encoding), resp.url or url)[:5]:
            text, att = _try_url(strategy, link, headers, source, get, min_articles)
            if text:
                return text, att
        return None, Attempt(strategy, url, False, "300 Multiple Choices, no usable item")
    if resp.status_code >= 400:
        return None, Attempt(strategy, url, False, f"HTTP {resp.status_code}")
    text = _response_text(resp)
    ok, detail = validate(text, source, min_articles)
    return (text if ok else None), Attempt(strategy, getattr(resp, "url", url) or url, ok,
                                           f"HTTP {resp.status_code}, {detail}")


def _saved_file(source: str) -> Path | None:
    """Find a browser-saved copy in data/raw/ regardless of the name the browser gave it."""
    pattern = SOURCES[source]["saved_name_hint"]
    if not config.RAW_DIR.exists():
        return None
    hits = [p for p in config.RAW_DIR.glob("*.htm*") if pattern in p.name]
    return max(hits, key=lambda p: p.stat().st_size) if hits else None


def _record(source: str, path: Path, strategy: str, url: str, detail: str) -> None:
    mpath = config.RAW_DIR / "manifest.json"
    manifest = json.loads(mpath.read_text()) if mpath.exists() else {}
    data = path.read_bytes()
    manifest[source] = {
        "strategy": strategy, "url": url, "detail": detail, "bytes": len(data),
        "sha256": hashlib.sha256(data).hexdigest(), "fetched_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
    }
    mpath.write_text(json.dumps(manifest, indent=2))


def fetch(source: str, refresh: bool = False, verbose: bool = False, get: Getter | None = None,
          min_articles: int | None = None) -> Path:
    """Return the path to a validated copy of `source`, downloading it if needed."""
    get = get or requests.get
    config.RAW_DIR.mkdir(parents=True, exist_ok=True)
    path = config.RAW_DIR / f"{source}.html"
    short = SOURCES[source]["short"]

    if path.exists() and not refresh:
        ok, detail = validate(path.read_text(encoding="utf-8", errors="replace"), source, min_articles)
        if ok:
            return path
        print(f"{short}: cached copy is invalid ({detail}), re-fetching")

    attempts: list[Attempt] = []
    for strategy, url, headers in _strategies(source):
        text, att = _try_url(strategy, url, headers, source, get, min_articles)
        attempts.append(att)
        if verbose or att.ok:
            print(f"{short}: [{'OK' if att.ok else '--'}] {strategy:19} {att.detail}  <{att.url}>")
        if text:
            tmp = path.with_suffix(".tmp")
            tmp.write_text(text, encoding="utf-8")
            tmp.replace(path)
            _record(source, path, strategy, att.url, att.detail)
            return path

    saved = _saved_file(source)
    if saved and saved != path:
        ok, detail = validate(saved.read_text(encoding="utf-8", errors="replace"), source, min_articles)
        attempts.append(Attempt("saved-file", str(saved), ok, detail))
        if ok:
            shutil.copyfile(saved, path)
            _record(source, path, "saved-file", str(saved), detail)
            print(f"{short}: [OK] saved-file          {detail}  <{saved.name}>")
            return path

    lines = "\n".join(f"  - {a.strategy}: {a.detail} <{a.url}>" for a in attempts)
    raise RuntimeError(
        f"Could not fetch {short} automatically. Attempts:\n{lines}\n"
        f"Last resort: open {SOURCES[source]['html_url']} in a browser, save it into {config.RAW_DIR} "
        f"(any filename) and re-run."
    )
