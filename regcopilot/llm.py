"""Model clients behind one interface, so the same pipeline runs on:

* "mistral" — Mistral La Plateforme (Mistral's hosted API)
* "ollama"  — open-weight Mistral models on your own hardware ("sovereign mode": no data leaves the machine)
* "mock"    — deterministic offline stand-in for tests and CI

Plain HTTP via `requests` keeps dependencies minimal and makes the API calls easy to explain.
"""
from __future__ import annotations

import hashlib
import re
import time
from dataclasses import dataclass

import numpy as np
import requests

from . import config


@dataclass
class ChatResult:
    text: str
    model: str
    prompt_tokens: int
    completion_tokens: int
    latency_s: float


def _post(url: str, payload: dict, headers: dict | None = None, timeout: int = 120, retries: int = 6) -> dict:
    delay = 1.0
    for attempt in range(retries):
        resp = requests.post(url, json=payload, headers=headers or {}, timeout=timeout)
        if resp.status_code in (429, 500, 502, 503, 504) and attempt < retries - 1:
            time.sleep(delay)
            delay = min(delay * 2, 30)
            continue
        if resp.status_code >= 400:
            raise RuntimeError(f"{url} returned {resp.status_code}: {resp.text[:500]}")
        return resp.json()
    raise RuntimeError(f"{url}: retries exhausted")


class MistralClient:
    name = "mistral"

    def __init__(self, api_key: str | None = None, base_url: str | None = None):
        self.api_key = api_key or config.MISTRAL_API_KEY
        if not self.api_key:
            raise RuntimeError("MISTRAL_API_KEY is not set. Add it to .env (see .env.example).")
        self.base_url = (base_url or config.MISTRAL_BASE_URL).rstrip("/")
        self.headers = {"Authorization": f"Bearer {self.api_key}", "Content-Type": "application/json"}
        self.default_chat_model = config.MISTRAL_CHAT_MODEL
        self.default_embed_model = config.MISTRAL_EMBED_MODEL

    def chat(self, messages: list[dict], model: str | None = None, temperature: float = 0.0,
             max_tokens: int = 700, json_mode: bool = False) -> ChatResult:
        model = model or self.default_chat_model
        payload = {"model": model, "messages": messages, "temperature": temperature, "max_tokens": max_tokens}
        if json_mode:
            payload["response_format"] = {"type": "json_object"}
        t0 = time.perf_counter()
        data = _post(f"{self.base_url}/chat/completions", payload, self.headers)
        usage = data.get("usage", {})
        return ChatResult(
            text=data["choices"][0]["message"]["content"],
            model=data.get("model", model),
            prompt_tokens=usage.get("prompt_tokens", 0),
            completion_tokens=usage.get("completion_tokens", 0),
            latency_s=time.perf_counter() - t0,
        )

    def embed(self, texts: list[str], model: str | None = None, batch_size: int = 32) -> np.ndarray:
        model = model or self.default_embed_model
        out: list[list[float]] = []
        for i in range(0, len(texts), batch_size):
            data = _post(f"{self.base_url}/embeddings", {"model": model, "input": texts[i:i + batch_size]}, self.headers)
            out.extend(d["embedding"] for d in sorted(data["data"], key=lambda d: d["index"]))
        return np.asarray(out, dtype=np.float32)


class OllamaClient:
    name = "ollama"

    def __init__(self, base_url: str | None = None):
        self.base_url = (base_url or config.OLLAMA_BASE_URL).rstrip("/")
        self.default_chat_model = config.OLLAMA_CHAT_MODEL
        self.default_embed_model = config.OLLAMA_EMBED_MODEL

    def chat(self, messages: list[dict], model: str | None = None, temperature: float = 0.0,
             max_tokens: int = 700, json_mode: bool = False) -> ChatResult:
        model = model or self.default_chat_model
        payload = {"model": model, "messages": messages, "stream": False,
                   "options": {"temperature": temperature, "num_predict": max_tokens, "num_ctx": 8192}}
        if json_mode:
            payload["format"] = "json"
        t0 = time.perf_counter()
        data = _post(f"{self.base_url}/api/chat", payload, timeout=600)
        return ChatResult(
            text=data["message"]["content"],
            model=model,
            prompt_tokens=data.get("prompt_eval_count", 0),
            completion_tokens=data.get("eval_count", 0),
            latency_s=time.perf_counter() - t0,
        )

    def embed(self, texts: list[str], model: str | None = None, batch_size: int = 32) -> np.ndarray:
        model = model or self.default_embed_model
        out: list[list[float]] = []
        for i in range(0, len(texts), batch_size):
            data = _post(f"{self.base_url}/api/embed", {"model": model, "input": texts[i:i + batch_size]}, timeout=600)
            out.extend(data["embeddings"])
        return np.asarray(out, dtype=np.float32)


class MockClient:
    """Offline stand-in. Embeddings are hashed bags of words; chat answers extractively from the
    first source in the prompt (or refuses if there is none). Good enough to test the plumbing."""
    name = "mock"
    default_chat_model = "mock-extractive"
    default_embed_model = "mock-hash"

    def chat(self, messages: list[dict], model: str | None = None, temperature: float = 0.0,
             max_tokens: int = 700, json_mode: bool = False) -> ChatResult:
        prompt = messages[-1]["content"]
        if json_mode:  # used by the judge
            text = '{"score": 3, "reason": "mock judge"}'
        else:
            m = re.search(r"\[SOURCE: ([^\]]+)\][^\n]*\n(.+)", prompt)
            if not m:
                from .rag import REFUSAL
                text = REFUSAL
            else:
                text = f"{m.group(2).strip()[:300]} [{m.group(1)}]"
        return ChatResult(text, model or self.default_chat_model, len(prompt.split()), len(text.split()), 0.0)

    def embed(self, texts: list[str], model: str | None = None, batch_size: int = 32, dim: int = 256) -> np.ndarray:
        from .bm25 import tokenize
        vecs = np.zeros((len(texts), dim), dtype=np.float32)
        for i, t in enumerate(texts):
            for tok in tokenize(t):
                vecs[i, int(hashlib.md5(tok.encode()).hexdigest(), 16) % dim] += 1.0
        return vecs


def get_client(backend: str):
    backend = backend.lower()
    if backend == "mistral":
        return MistralClient()
    if backend == "ollama":
        return OllamaClient()
    if backend == "mock":
        return MockClient()
    raise ValueError(f"Unknown backend '{backend}' (use mistral, ollama or mock)")
