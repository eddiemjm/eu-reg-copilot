"""Look and feel: CSS and the compact page header.

Everything here is served by the app itself (no web fonts, no CDNs), so sovereign mode really
does keep every request on the machine.
"""
from __future__ import annotations

import html

import streamlit as st

# Chart series colours, validated for colour-vision deficiency and contrast on the dark surface.
SERIES = {"mistral-large-latest": "#E25C0C", "mistral-small-latest": "#5585E0"}
PIXELS = ["#FFD800", "#FFAF00", "#FF8205", "#FA500F", "#FFAF00", "#FF8205", "#FA500F", "#E10500"]

CSS = """
<style>
:root {
  --rc-bg: #0E0E10; --rc-surface: #17171B; --rc-surface-2: #1F1F25;
  --rc-line: rgba(255, 255, 255, 0.08); --rc-line-strong: rgba(255, 255, 255, 0.16);
  --rc-ink: #ECECEF; --rc-muted: #9A9AA6; --rc-faint: #6B6B76;
  --rc-accent: #FA500F;
  --rc-grad: linear-gradient(90deg, #E10500 0%, #FA500F 30%, #FF8205 55%, #FFAF00 80%, #FFD800 100%);
  --rc-ok: #8FE3B5; --rc-ok-bg: rgba(63, 185, 122, 0.12); --rc-ok-line: rgba(63, 185, 122, 0.45);
  --rc-warn: #F5C451; --rc-warn-line: rgba(242, 178, 51, 0.65);
  --rc-bad: #FF8A85; --rc-bad-bg: rgba(240, 82, 77, 0.12); --rc-bad-line: rgba(240, 82, 77, 0.6);
  --rc-mono: ui-monospace, "SF Mono", Menlo, Consolas, monospace;
}
.block-container { padding-top: 3.2rem; padding-bottom: 6rem; max-width: 1400px; }
header[data-testid="stHeader"] { background: rgba(14, 14, 16, 0.85); backdrop-filter: blur(8px); }

/* ---------- header ---------- */
.rc-header { display: flex; align-items: center; justify-content: space-between; gap: 1.5rem;
  flex-wrap: wrap; padding: 0.6rem 0 0.9rem; }
.rc-brand { display: flex; align-items: center; gap: 0.9rem; }
.rc-pixels { display: grid; grid-template-columns: repeat(4, 9px); grid-auto-rows: 9px; gap: 2px; }
.rc-pixels i { display: block; border-radius: 1px; }
.rc-title { font-size: 1.45rem; font-weight: 700; letter-spacing: -0.01em; color: var(--rc-ink); line-height: 1.2; }
.rc-sub { color: var(--rc-muted); font-size: 0.88rem; }
.rc-kpis { display: flex; flex-direction: column; align-items: flex-end; gap: 0.25rem; margin-left: auto; }
.rc-kpi-row { display: flex; gap: 1.6rem; }
.rc-kpi b { display: block; font-size: 1.15rem; font-weight: 700; color: var(--rc-ink);
  font-variant-numeric: tabular-nums; }
.rc-kpi span { color: var(--rc-muted); font-size: 0.72rem; text-transform: uppercase; letter-spacing: 0.06em; }
.rc-kpi-src { color: var(--rc-faint); font-size: 0.72rem; font-family: var(--rc-mono); }
.rc-mode { border: 1px solid var(--rc-line-strong); border-radius: 999px; padding: 0.1rem 0.6rem;
  font-size: 0.75rem; color: var(--rc-muted); white-space: nowrap; margin-left: 0.6rem; vertical-align: 0.2rem;
  font-weight: 500; letter-spacing: 0; }
.rc-mode.local { border-color: var(--rc-ok-line); color: var(--rc-ok); }
.rc-rule { height: 2px; background: var(--rc-grad); border-radius: 2px; margin-bottom: 1.2rem; opacity: 0.9; }

/* ---------- example cards ---------- */
.rc-hint { color: var(--rc-muted); font-size: 0.9rem; margin: 0.4rem 0 0.6rem; }
[class*="st-key-ex-"] button { min-height: 6.2rem; height: 100%; align-items: flex-start; justify-content: flex-start;
  text-align: left; background: var(--rc-surface); border: 1px solid var(--rc-line); border-radius: 12px;
  padding: 0.85rem 1rem; transition: border-color 120ms ease, transform 120ms ease; }
[class*="st-key-ex-"] button:hover { border-color: var(--rc-accent); transform: translateY(-1px); }
[class*="st-key-ex-"] button p { text-align: left; font-size: 0.93rem; line-height: 1.45; }
[class*="st-key-ex-"] button div, [class*="st-key-ex-"] button p { white-space: normal; text-overflow: clip; }

/* ---------- citation pills ---------- */
.rc-pill { display: inline-block; border-radius: 6px; padding: 0 0.4rem; margin: 0 0.1rem;
  font-size: 0.82em; font-family: var(--rc-mono); white-space: nowrap; cursor: help; line-height: 1.55; }
.rc-pill.ok { background: var(--rc-ok-bg); border: 1px solid var(--rc-ok-line); color: var(--rc-ok); }
.rc-pill.missing { border: 1px dashed var(--rc-warn-line); color: var(--rc-warn); }
.rc-pill.invented { background: var(--rc-bad-bg); border: 1px solid var(--rc-bad-line); color: var(--rc-bad);
  text-decoration: line-through; }

/* ---------- grounding badge ---------- */
.rc-badge { display: flex; flex-wrap: wrap; gap: 0.4rem 1rem; align-items: center; margin-top: 0.6rem;
  font-size: 0.8rem; color: var(--rc-muted); font-family: var(--rc-mono); }
.rc-badge .lead { font-family: inherit; font-weight: 600; padding: 0.1rem 0.55rem; border-radius: 6px; }
.rc-badge .lead.ok { color: var(--rc-ok); background: var(--rc-ok-bg); }
.rc-badge .lead.warn { color: var(--rc-warn); border: 1px dashed var(--rc-warn-line); }
.rc-badge .lead.bad { color: var(--rc-bad); background: var(--rc-bad-bg); }
.rc-badge .lead.none { color: var(--rc-muted); border: 1px solid var(--rc-line-strong); }

/* ---------- evidence ---------- */
.rc-evi-title { font-size: 0.72rem; text-transform: uppercase; letter-spacing: 0.08em; color: var(--rc-muted);
  margin-bottom: 0.4rem; }
[class*="st-key-evi-"] { background: var(--rc-surface); border: 1px solid var(--rc-line); border-radius: 10px;
  padding: 0.6rem 0.8rem 0.35rem; gap: 0.5rem; }
[class*="st-key-evi-"][class*="-cited"] { border-left: 3px solid var(--rc-accent); }
[class*="st-key-evi-"] [data-testid="stMarkdownContainer"] { padding-bottom: 0.15rem; }
[class*="st-key-evi-"] [data-testid="stExpander"] details { border: none; background: transparent; }
[class*="st-key-evi-"] [data-testid="stExpander"] summary { padding: 0.1rem 0; font-size: 0.82rem; color: var(--rc-muted); }
[class*="st-key-evi-"] [data-testid="stExpander"] summary:hover { color: var(--rc-ink); }
.rc-evi-head { font-size: 0.9rem; color: var(--rc-ink); }
.rc-evi-head b { font-weight: 700; }
.rc-evi-meta { font-family: var(--rc-mono); font-size: 0.72rem; color: var(--rc-faint); margin-top: 0.3rem; }
.rc-evi-meta .cited { color: var(--rc-accent); font-weight: 600; }
.rc-rank { display: inline-block; border: 1px solid var(--rc-line-strong); border-radius: 4px; padding: 0 0.3rem;
  margin-right: 0.25rem; }

/* ---------- cards (how it works / setup) ---------- */
.rc-card { background: var(--rc-surface); border: 1px solid var(--rc-line); border-radius: 12px;
  padding: 0.9rem 1rem; height: 100%; }
.rc-card h4 { margin: 0 0 0.3rem; font-size: 0.98rem; color: var(--rc-ink); }
.rc-card p { margin: 0; color: var(--rc-muted); font-size: 0.87rem; line-height: 1.45; }
.rc-verdict { background: var(--rc-surface); border: 1px solid var(--rc-line); border-left: 3px solid var(--rc-accent);
  border-radius: 10px; padding: 0.7rem 1rem; color: var(--rc-ink); font-size: 0.92rem; }
.rc-note { background: var(--rc-surface); border: 1px solid var(--rc-line); border-radius: 10px; padding: 0.8rem 1rem;
  color: var(--rc-muted); font-size: 0.88rem; line-height: 1.5; margin: 0.6rem 0 1rem; }
.rc-note b { color: var(--rc-ink); }
.rc-col-head { font-family: var(--rc-mono); font-size: 0.85rem; color: var(--rc-muted); margin-bottom: 0.3rem; }
.rc-col-head b { color: var(--rc-ink); }
.rc-dot { display: inline-block; width: 9px; height: 9px; border-radius: 2px; margin-right: 0.4rem; }
</style>
"""


def favicon():
    """The pixel motif as a PIL image. Emoji favicons are fetched from a CDN, which sovereign mode can't allow."""
    from PIL import Image, ImageDraw
    img = Image.new("RGBA", (64, 64), (0, 0, 0, 0))
    draw = ImageDraw.Draw(img)
    for i, colour in enumerate(PIXELS):
        x, y = (i % 4) * 16, 16 + (i // 4) * 16
        draw.rectangle([x + 1, y + 1, x + 14, y + 14], fill=colour)
    return img


def inject() -> None:
    st.markdown(CSS, unsafe_allow_html=True)


def header(mode_label: str, local: bool, kpis: list[tuple[str, str]], kpi_source: str = "") -> None:
    pixels = "".join(f'<i style="background:{c}"></i>' for c in PIXELS)
    items = "".join(f'<div class="rc-kpi"><b>{html.escape(v)}</b><span>{html.escape(k)}</span></div>'
                    for k, v in kpis)
    src = f'<div class="rc-kpi-src">{html.escape(kpi_source)}</div>' if kpi_source else ""
    st.markdown(
        f'<div class="rc-header">'
        f'<div class="rc-brand"><div class="rc-pixels">{pixels}</div><div>'
        f'<div class="rc-title">EU AI Act &amp; DORA Copilot'
        f'<span class="rc-mode{" local" if local else ""}">{html.escape(mode_label)}</span></div>'
        f'<div class="rc-sub">Answers grounded in the official texts, every citation checked · Built on Mistral models</div>'
        f'</div></div>'
        f'<div class="rc-kpis"><div class="rc-kpi-row">{items}</div>{src}</div>'
        f'</div><div class="rc-rule"></div>',
        unsafe_allow_html=True)
