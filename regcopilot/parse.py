"""Turn EUR-Lex HTML into structure-aware chunks.

Legal text has structure that generic chunkers destroy: an answer about fines lives in one
Article, and a lawyer expects a citation to that Article (and paragraph), not to "chunk 412".
So we parse the regulation into its legal units (Articles, Annexes, optional Recitals) and
chunk *within* each unit, carrying the citation label with every chunk.

The parser works on text lines rather than CSS class names, so it is robust to EUR-Lex
markup changes: it only relies on headings such as "Article 27", "ANNEX III", "CHAPTER V".
"""
from __future__ import annotations

import re
from dataclasses import asdict, dataclass, field

from bs4 import BeautifulSoup

from .sources import SOURCES

RE_ARTICLE = re.compile(r"^Article\s+(\d+[a-z]?)$")
RE_ANNEX = re.compile(r"^ANNEX\s+([IVXLC]+)$")
RE_CHAPTER = re.compile(r"^CHAPTER\s+([IVXLC]+)$")
RE_SECTION = re.compile(r"^SECTION\s+(\d+|[IVXLC]+)$", re.I)
RE_RECITAL = re.compile(r"^\(\s*(\d+)\s*\)\s+(.+)$")
RE_PARA = re.compile(r"^(\d+)\.\s+")
RE_FOOTNOTE = re.compile(
    r"^\(\s*\d+\s*\)\s*(OJ|Regulation|Directive|Council|Commission|Decision|Position|Recommendation|Judgment)\b"
)


@dataclass
class Unit:
    source: str            # "AIA" | "DORA"
    kind: str              # "article" | "annex" | "recital"
    number: str            # "27", "III", "42"
    title: str = ""
    chapter: str = ""
    chapter_title: str = ""
    lines: list[str] = field(default_factory=list)

    @property
    def unit_id(self) -> str:
        tag = {"article": "Art", "annex": "Annex", "recital": "Rec"}[self.kind]
        return f"{self.source}-{tag}-{self.number}"

    @property
    def label(self) -> str:
        short = SOURCES[self.source]["short"]
        tag = {"article": "Art.", "annex": "Annex", "recital": "Recital"}[self.kind]
        return f"{short} {tag} {self.number}"


@dataclass
class Chunk:
    chunk_id: str
    unit_id: str
    label: str              # citation label, e.g. "AI Act Art. 27"
    source: str
    kind: str
    number: str
    title: str
    chapter: str
    paragraphs: list[str]   # paragraph numbers covered, e.g. ["1", "2"]
    text: str               # header + body; this is what gets embedded and shown

    def to_dict(self) -> dict:
        return asdict(self)


def _norm(text: str) -> str:
    text = text.replace("\xa0", " ").replace(" ", " ").replace(" ", " ")
    return re.sub(r"\s+", " ", text).strip()


def extract_lines(html: str) -> list[str]:
    """Flatten the document into ordered text lines. Table rows (EUR-Lex uses tables for
    numbered points and recitals) become a single line, e.g. "(a) the placing on the market..."."""
    soup = BeautifulSoup(html, "lxml")
    for tag in soup(["script", "style", "noscript", "head"]):
        tag.decompose()
    body = soup.body or soup
    lines: list[str] = []
    for el in body.find_all(["p", "tr", "h1", "h2", "h3", "h4", "li"]):
        if el.find_parent("tr") is not None:
            continue  # content already captured by its (outermost) row
        if el.name in ("p", "li") and el.find_parent("li") is not None:
            continue
        if el.name == "tr":
            cells = el.find_all(["td", "th"], recursive=False)
            text = " ".join(c.get_text(" ", strip=True) for c in cells)
        else:
            text = el.get_text(" ", strip=True)
        text = _norm(text)
        if text:
            lines.append(text)
    return lines


def parse_units(html: str, source: str, include_recitals: bool = False) -> list[Unit]:
    lines = extract_lines(html)
    units: list[Unit] = []
    current: Unit | None = None
    chapter, chapter_title = "", ""
    expect_title = False           # next line after a heading is its title
    expect_chapter_title = False
    skip_next = False              # section titles are structural, not article text
    in_preamble = True             # recitals live before "HAVE ADOPTED THIS REGULATION"
    after_signature = False        # skip the "Done at ..." signature block and footnotes

    def close():
        nonlocal current
        if current is not None and (current.lines or current.title):
            units.append(current)
        current = None

    for line in lines:
        if in_preamble:
            if line.upper().startswith("HAVE ADOPTED THIS REGULATION"):
                close()
                in_preamble = False
                continue
            m = RE_RECITAL.match(line)
            if m and include_recitals:
                close()
                current = Unit(source, "recital", m.group(1), title="", lines=[m.group(2)])
            elif current is not None and current.kind == "recital" and not RE_ARTICLE.match(line):
                current.lines.append(line)
            if not RE_ARTICLE.match(line):
                continue
            in_preamble = False  # documents without the formula: fall through

        if after_signature and RE_FOOTNOTE.match(line):
            continue

        if skip_next:
            skip_next = False
            if not RE_ARTICLE.match(line):
                continue
        if RE_SECTION.match(line):
            skip_next = True
            continue

        if m := RE_CHAPTER.match(line):
            chapter, chapter_title = m.group(1), ""
            expect_chapter_title = True
            continue
        if expect_chapter_title:
            expect_chapter_title = False
            if not RE_ARTICLE.match(line) and not RE_SECTION.match(line):
                chapter_title = line
                continue

        if m := RE_ARTICLE.match(line):
            if after_signature:
                continue  # stray reference inside annexes/footnotes
            close()
            current = Unit(source, "article", m.group(1), chapter=chapter, chapter_title=chapter_title)
            expect_title = True
            continue
        if m := RE_ANNEX.match(line):
            close()
            current = Unit(source, "annex", m.group(1))
            expect_title = True
            continue

        if line.startswith("Done at "):
            close()
            after_signature = True
            continue
        if after_signature and current is None:
            continue  # signatures ("For the European Parliament", names...)

        if current is None:
            continue
        if expect_title:
            expect_title = False
            if not RE_PARA.match(line) and len(line) < 250:
                current.title = line
                continue
        current.lines.append(line)

    close()
    # De-duplicate (some EUR-Lex pages repeat headings in a table of contents).
    seen: dict[str, Unit] = {}
    for u in units:
        if u.unit_id not in seen or len(" ".join(u.lines)) > len(" ".join(seen[u.unit_id].lines)):
            seen[u.unit_id] = u
    return list(seen.values())


def _paragraphs(unit: Unit) -> list[tuple[str, list[str]]]:
    """Group a unit's lines into numbered paragraphs ("1.", "2." ...)."""
    groups: list[tuple[str, list[str]]] = []
    for line in unit.lines:
        m = RE_PARA.match(line)
        if m or not groups:
            groups.append((m.group(1) if m else "", [line]))
        else:
            groups[-1][1].append(line)
    return groups


def chunk_units(units: list[Unit], max_words: int = 350) -> list[Chunk]:
    chunks: list[Chunk] = []
    for u in units:
        header = f"{u.label}" + (f" — {u.title}" if u.title else "")
        if u.chapter:
            header += f" (Chapter {u.chapter}{': ' + u.chapter_title if u.chapter_title else ''})"
        buckets: list[tuple[list[str], list[str]]] = []  # (para numbers, lines)
        words = 0
        for num, plines in _paragraphs(u):
            pwords = sum(len(l.split()) for l in plines)
            if pwords > max_words:  # very long paragraph: split by line
                for line in plines:
                    lw = len(line.split())
                    if not buckets or words + lw > max_words:
                        buckets.append(([num] if num else [], []))
                        words = 0
                    if num and num not in buckets[-1][0]:
                        buckets[-1][0].append(num)
                    buckets[-1][1].append(line)
                    words += lw
                continue
            if not buckets or words + pwords > max_words:
                buckets.append(([], []))
                words = 0
            if num:
                buckets[-1][0].append(num)
            buckets[-1][1].extend(plines)
            words += pwords
        if not buckets:
            buckets = [([], [])]
        for i, (nums, blines) in enumerate(buckets):
            chunks.append(Chunk(
                chunk_id=f"{u.unit_id}#{i}",
                unit_id=u.unit_id,
                label=u.label,
                source=u.source,
                kind=u.kind,
                number=u.number,
                title=u.title,
                chapter=u.chapter,
                paragraphs=nums,
                text=header + "\n" + "\n".join(blines),
            ))
    return chunks
