"""The regulations in the corpus. Both are public EU law, published on EUR-Lex."""

SOURCES = {
    "AIA": {
        "short": "AI Act",
        "title": "Regulation (EU) 2024/1689 laying down harmonised rules on artificial intelligence (Artificial Intelligence Act)",
        "celex": "32024R1689",
        "html_url": "https://eur-lex.europa.eu/legal-content/EN/TXT/HTML/?uri=CELEX:32024R1689",
        "eli_url": "https://eur-lex.europa.eu/eli/reg/2024/1689/oj",
        # File name of the English XHTML in the Official Journal manifestation on Cellar
        "cellar_item": "L_202401689EN.000101.fmx.xml.html",
        "saved_name_hint": "202401689",   # browsers save the page as L_202401689EN...html
        "min_articles": 100,              # the AI Act has 113 Articles
    },
    "DORA": {
        "short": "DORA",
        "title": "Regulation (EU) 2022/2554 on digital operational resilience for the financial sector (DORA)",
        "celex": "32022R2554",
        "html_url": "https://eur-lex.europa.eu/legal-content/EN/TXT/HTML/?uri=CELEX:32022R2554",
        "eli_url": "https://eur-lex.europa.eu/eli/reg/2022/2554/oj",
        "cellar_item": "L_2022333EN.01000101.xml.html",
        "saved_name_hint": "2022333",     # OJ L 333, 27.12.2022
        "min_articles": 60,               # DORA has 64 Articles
    },
}

# Map the short names a model might write in a citation back to source keys.
SHORT_TO_KEY = {v["short"].lower(): k for k, v in SOURCES.items()}
