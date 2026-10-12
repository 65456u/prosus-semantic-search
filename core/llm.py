"""Cached structured-output LLM access shared by query understanding and evaluation.

Models are named with a provider prefix:

    ollama/qwen3.5:9b-q4_K_M      local Ollama server (default http://127.0.0.1:11434)
    openai/gpt-4.1-mini           OpenAI API; the key comes from the environment or a Git-ignored `.env`

Every request is keyed by its exact payload, so reruns are free and
deterministic. Each call that reaches a model is appended to a usage ledger
with token counts, wall time, and an estimated cost (zero for local models).
"""

from __future__ import annotations

import hashlib
import json
import os
import sqlite3
import threading
import time
from pathlib import Path
from typing import Any

import httpx

ROOT = Path(__file__).resolve().parents[1]
CACHE_DIR = ROOT / ".demo-cache" / "llm"
OLLAMA_URL = os.environ.get("OLLAMA_URL", "http://127.0.0.1:11434")

# USD per million tokens (input, output), OpenAI list prices used only for
# estimates. The OpenAI usage dashboard is the source of truth.
PRICES = {
    "openai/gpt-4.1": (2.00, 8.00),
    "openai/gpt-4.1-mini": (0.40, 1.60),
    "openai/gpt-4.1-nano": (0.10, 0.40),
}


def load_env(path: Path = ROOT / ".env") -> None:
    """Minimal `.env` support; existing environment variables take precedence."""
    if not path.is_file():
        return
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        os.environ.setdefault(key.strip(), value.strip().strip("'\""))


def _digest(value: Any) -> str:
    return hashlib.sha256(json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def split_model(model: str) -> tuple[str, str]:
    provider, _, name = model.partition("/")
    if provider not in ("ollama", "openai") or not name:
        raise ValueError(f"Model must be 'ollama/<tag>' or 'openai/<name>', got {model!r}.")
    return provider, name


def estimate_cost(model: str, input_tokens: int, output_tokens: int = 0) -> float:
    price_in, price_out = PRICES.get(model, (0.0, 0.0))
    return (input_tokens * price_in + output_tokens * price_out) / 1_000_000


class LLM:
    """Thread-safe cached wrapper around structured chat completions."""

    def __init__(self, cache_dir: str | Path = CACHE_DIR, offline: bool = False):
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.offline = offline
        self._openai = None
        self._client_lock = threading.Lock()
        self._ledger_lock = threading.Lock()
        self._http = httpx.Client(base_url=OLLAMA_URL, timeout=httpx.Timeout(900.0, connect=10.0))
        with sqlite3.connect(self._db, timeout=60) as connection:
            connection.execute("PRAGMA journal_mode=WAL")
            connection.execute("CREATE TABLE IF NOT EXISTS responses (key TEXT PRIMARY KEY, value TEXT NOT NULL)")

    @property
    def _db(self) -> Path:
        return self.cache_dir / "cache.sqlite3"

    @property
    def openai(self):
        with self._client_lock:
            if self._openai is None:
                load_env()
                if not os.environ.get("OPENAI_API_KEY"):
                    raise RuntimeError("OPENAI_API_KEY is not set. Put it in the environment or in a local .env file.")
                from openai import OpenAI
                self._openai = OpenAI(max_retries=6, timeout=120)
            return self._openai

    def _log(self, task: str, model: str, input_tokens: int, output_tokens: int, seconds: float) -> None:
        record = {"time": time.strftime("%Y-%m-%dT%H:%M:%S"), "task": task, "model": model,
                  "input_tokens": input_tokens, "output_tokens": output_tokens, "seconds": round(seconds, 3),
                  "cost_usd": estimate_cost(model, input_tokens, output_tokens)}
        with self._ledger_lock, (self.cache_dir / "usage.jsonl").open("a", encoding="utf-8") as ledger:
            ledger.write(json.dumps(record) + "\n")

    def json(self, task: str, model: str, system: str, user: str, schema: dict[str, Any],
             temperature: float = 0.0, seed: int = 7, use_cache: bool = True,
             images: list[str] | None = None) -> dict[str, Any]:
        """Schema-constrained chat completion with reasoning disabled, returned as parsed JSON.

        `use_cache=False` neither reads nor writes the response cache (for latency benchmarks).
        `images` are base64-encoded pictures attached to the user message (Ollama vision models);
        they are part of the cache key.
        """
        provider, name = split_model(model)
        user_message: dict[str, Any] = {"role": "user", "content": user}
        if images:
            if provider != "ollama":
                raise ValueError("Image inputs are only wired for Ollama models.")
            user_message["images"] = images
        request = {"provider": provider, "model": name, "temperature": temperature, "seed": seed,
                   "messages": [{"role": "system", "content": system}, user_message],
                   "schema": schema}
        key = _digest({"kind": "chat", **request})
        if use_cache:
            with sqlite3.connect(self._db, timeout=60) as connection:
                row = connection.execute("SELECT value FROM responses WHERE key=?", (key,)).fetchone()
            if row:
                return json.loads(row[0])
        if self.offline:
            raise RuntimeError("Offline mode: the request is not in the local cache.")
        start = time.perf_counter()
        if provider == "ollama":
            parsed, input_tokens, output_tokens = self._ollama(task, request)
        else:
            parsed, input_tokens, output_tokens = self._openai_chat(task, request)
        self._log(task, model, input_tokens, output_tokens, time.perf_counter() - start)
        if use_cache:
            with sqlite3.connect(self._db, timeout=60) as connection:
                connection.execute("INSERT OR REPLACE INTO responses(key,value) VALUES (?,?)", (key, json.dumps(parsed, ensure_ascii=False)))
        return parsed

    def _ollama(self, task: str, request: dict[str, Any]) -> tuple[dict[str, Any], int, int]:
        # `format` constrains decoding to the JSON schema; `think: false` disables reasoning traces.
        payload = {"model": request["model"], "messages": request["messages"], "format": request["schema"],
                   "stream": False, "think": False, "keep_alive": "1h",
                   "options": {"temperature": request["temperature"], "seed": request["seed"], "num_ctx": 4096}}
        last_error: Exception | None = None
        for attempt in range(4):
            try:
                response = self._http.post("/api/chat", json=payload)
                response.raise_for_status()
                body = response.json()
                parsed = json.loads(body["message"]["content"])
                return parsed, int(body.get("prompt_eval_count") or 0), int(body.get("eval_count") or 0)
            except (httpx.HTTPError, json.JSONDecodeError, KeyError) as error:
                last_error = error
                time.sleep(2 ** attempt)
        raise RuntimeError(f"Ollama request for {task} failed after retries: {last_error}")

    def _openai_chat(self, task: str, request: dict[str, Any]) -> tuple[dict[str, Any], int, int]:
        response = self.openai.chat.completions.create(
            model=request["model"], temperature=request["temperature"], seed=request["seed"], messages=request["messages"],
            response_format={"type": "json_schema", "json_schema": {"name": task, "strict": True, "schema": request["schema"]}},
        )
        usage = response.usage
        return json.loads(response.choices[0].message.content), usage.prompt_tokens, usage.completion_tokens

    def unload(self, model: str) -> None:
        """Free a local model's memory before loading another large model."""
        provider, name = split_model(model)
        if provider == "ollama" and not self.offline:
            self._http.post("/api/generate", json={"model": name, "keep_alive": 0})

    def describe(self, model: str) -> dict[str, Any]:
        """Exact local model identity (digest, quantization) for the report."""
        provider, name = split_model(model)
        if provider != "ollama":
            return {"model": model}
        try:
            tags = self._http.get("/api/tags").json().get("models", [])
        except httpx.HTTPError:
            return {"model": model}
        for entry in tags:
            if entry.get("name") == name or entry.get("model") == name:
                details = entry.get("details") or {}
                return {"model": model, "digest": entry.get("digest"), "parameter_size": details.get("parameter_size"),
                        "quantization": details.get("quantization_level")}
        return {"model": model}


def usage_summary(cache_dir: str | Path = CACHE_DIR) -> dict[str, Any]:
    """Aggregate the ledger of model calls that were actually made (cache misses)."""
    path = Path(cache_dir) / "usage.jsonl"
    summary: dict[str, dict[str, float]] = {}
    if path.is_file():
        for line in path.read_text(encoding="utf-8").splitlines():
            record = json.loads(line)
            entry = summary.setdefault(f"{record['task']}::{record['model']}",
                                       {"calls": 0, "input_tokens": 0, "output_tokens": 0, "seconds": 0.0, "cost_usd": 0.0})
            entry["calls"] += 1
            entry["input_tokens"] += record["input_tokens"]
            entry["output_tokens"] += record["output_tokens"]
            entry["seconds"] += record.get("seconds", 0.0)
            entry["cost_usd"] += record["cost_usd"]
    total = sum(entry["cost_usd"] for entry in summary.values())
    return {"by_task": summary, "total_cost_usd": round(total, 4)}
