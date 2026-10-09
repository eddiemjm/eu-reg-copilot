"""Central configuration. Values can be overridden with environment variables or a .env file."""
from __future__ import annotations

import json
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data"
RAW_DIR = DATA_DIR / "raw"
CHUNKS_PATH = DATA_DIR / "chunks.json"
INDEX_DIR = DATA_DIR / "index"
RESULTS_DIR = ROOT / "results"


def _load_dotenv(path: Path = ROOT / ".env") -> None:
    """Minimal .env loader (avoids an extra dependency). Never overrides real env vars."""
    if not path.exists():
        return
    for line in path.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        os.environ.setdefault(key.strip(), value.strip().strip('"').strip("'"))


_load_dotenv()

# --- Mistral La Plateforme (hosted API) ---
MISTRAL_API_KEY = os.getenv("MISTRAL_API_KEY", "")
MISTRAL_BASE_URL = os.getenv("MISTRAL_BASE_URL", "https://api.mistral.ai/v1")
MISTRAL_CHAT_MODEL = os.getenv("MISTRAL_CHAT_MODEL", "mistral-small-latest")
MISTRAL_EMBED_MODEL = os.getenv("MISTRAL_EMBED_MODEL", "mistral-embed")

# --- Local / sovereign mode (Ollama running open-weight models on your own machine) ---
OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
OLLAMA_CHAT_MODEL = os.getenv("OLLAMA_CHAT_MODEL", "mistral-nemo")
OLLAMA_EMBED_MODEL = os.getenv("OLLAMA_EMBED_MODEL", "nomic-embed-text")

# --- Retrieval ---
TOP_K = int(os.getenv("TOP_K", "6"))
MAX_CHUNK_WORDS = int(os.getenv("MAX_CHUNK_WORDS", "350"))

# --- Optional pricing for cost reporting in evals ---
# Fill prices.json with {"model-name": {"input_per_m": x, "output_per_m": y}} using current
# prices from https://mistral.ai/pricing. If a model is missing, cost is reported as n/a.
PRICES_PATH = ROOT / "prices.json"


def load_prices() -> dict:
    if PRICES_PATH.exists():
        return json.loads(PRICES_PATH.read_text())
    return {}
