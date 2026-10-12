"""Local Hugging Face retrieval models: an instruction-prompted dense encoder and a cross-encoder reranker.

Both load pinned snapshots from `.demo-cache/models/` (see core/download_model.py)
and never download at inference time. Vectors and reranker scores are cached on
disk, keyed by the model revision and the exact input, so reruns reproduce the
same rankings without recomputation.
"""

from __future__ import annotations

import hashlib
import json
import os
import sqlite3
import threading
from pathlib import Path
from typing import Any, Sequence

import numpy as np

# PyTorch's MPS backend fails (Metal command-buffer assertion) when two threads run
# inference at once, even on different models. Every local model, including e5 in
# retrieval.py, serializes inference through this one lock.
INFERENCE_LOCK = threading.Lock()


def pick_device(requested: str = "auto") -> str:
    import torch

    requested = os.environ.get("PROSUS_DEVICE", requested)
    if requested != "auto":
        return requested
    return "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"


def _digest(value: Any) -> str:
    return hashlib.sha256(json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def _manifest(model_dir: Path) -> dict[str, str]:
    path = model_dir / "prosus-model.json"
    if not path.is_file():
        raise FileNotFoundError(f"{model_dir} is missing; run `.venv/bin/python core/download_model.py`.")
    return json.loads(path.read_text())


class _ArrayStore:
    """Small SQLite key-value store for float32 arrays."""

    def __init__(self, path: Path):
        self.path = path
        path.parent.mkdir(parents=True, exist_ok=True)
        with sqlite3.connect(path, timeout=60) as connection:
            connection.execute("PRAGMA journal_mode=WAL")
            connection.execute("CREATE TABLE IF NOT EXISTS arrays (key TEXT PRIMARY KEY, value BLOB NOT NULL)")

    def get(self, keys: Sequence[str]) -> dict[str, np.ndarray]:
        found = {}
        with sqlite3.connect(self.path, timeout=60) as connection:
            for start in range(0, len(keys), 400):
                batch = list(keys[start:start + 400])
                rows = connection.execute(f"SELECT key,value FROM arrays WHERE key IN ({','.join('?' for _ in batch)})", batch)
                found.update((key, np.frombuffer(blob, dtype=np.float32).copy()) for key, blob in rows)
        return found

    def put(self, rows: Sequence[tuple[str, np.ndarray]]) -> None:
        with sqlite3.connect(self.path, timeout=60) as connection:
            connection.executemany("INSERT OR REPLACE INTO arrays(key,value) VALUES (?,?)",
                                   [(key, np.asarray(value, dtype=np.float32).tobytes()) for key, value in rows])


class DenseEncoder:
    """Sentence-transformers retriever with an instruction prompt on queries only.

    harrier-oss-v1 expects queries as "Instruct: {task}\\nQuery: {query}" and
    plain documents; embeddings are L2-normalized, so dot product is cosine.
    """

    def __init__(self, model_dir: str | Path, cache_dir: str | Path, query_instruction: str,
                 device: str = "auto", batch_size: int = 32, max_seq_length: int = 512):
        self.model_dir = Path(model_dir)
        self.cache_dir = Path(cache_dir)
        manifest = _manifest(self.model_dir)
        self.query_prompt = f"Instruct: {query_instruction}\nQuery: "
        self.identity = {"repo_id": manifest["repo_id"], "revision": manifest["revision"],
                         "query_prompt": self.query_prompt, "max_seq_length": max_seq_length, "dtype": "float32"}
        self.name = manifest["repo_id"]
        self.requested_device = device
        self.batch_size = batch_size
        self.max_seq_length = max_seq_length
        self._model = None
        self._queries = _ArrayStore(self.cache_dir / "dense-queries.sqlite3")

    def _encode(self, texts: list[str], prompt: str | None) -> np.ndarray:
        with INFERENCE_LOCK:
            if self._model is None:
                from sentence_transformers import SentenceTransformer
                self._model = SentenceTransformer(str(self.model_dir), device=pick_device(self.requested_device), local_files_only=True)
                self._model.max_seq_length = self.max_seq_length
            vectors = self._model.encode(texts, prompt=prompt, batch_size=self.batch_size, normalize_embeddings=True,
                                         convert_to_numpy=True, show_progress_bar=False)
        vectors = np.asarray(vectors, dtype=np.float32)
        if vectors.shape[0] != len(texts) or not np.isfinite(vectors).all():
            raise ValueError("Encoder returned invalid embeddings.")
        return vectors

    def documents(self, texts: list[str]) -> np.ndarray:
        """Document matrix, cached as one .npy file per exact corpus and model identity."""
        path = self.cache_dir / "dense-docs" / f"{_digest({**self.identity, 'documents': texts})}.npy"
        if path.is_file():
            matrix = np.load(path, allow_pickle=False)
            if matrix.shape[0] == len(texts):
                return matrix
        matrix = self._encode(texts, prompt=None)
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary = path.with_suffix(".npy.tmp")
        with temporary.open("wb") as stream:
            np.save(stream, matrix, allow_pickle=False)
        os.replace(temporary, path)
        return matrix

    def queries(self, texts: list[str]) -> np.ndarray:
        keys = [_digest({**self.identity, "query": text}) for text in texts]
        known = self._queries.get(keys)
        missing = [index for index, key in enumerate(keys) if key not in known]
        if missing:
            encoded = self._encode([texts[index] for index in missing], prompt=self.query_prompt)
            rows = [(keys[index], vector) for index, vector in zip(missing, encoded)]
            self._queries.put(rows)
            known.update(rows)
        return np.stack([known[key] for key in keys])


class CrossEncoderReranker:
    """llama-nemotron-rerank-1b-v2: one relevance logit per (query, passage) pair.

    Input format follows the model card: "question:{q} \\n \\n passage:{p}".
    The model code ships with the pinned snapshot (trust_remote_code) and was
    reviewed: it only defines a bidirectional Llama classifier.
    """

    def __init__(self, model_dir: str | Path, cache_dir: str | Path, device: str = "auto",
                 batch_size: int = 16, max_length: int = 512):
        self.model_dir = Path(model_dir)
        manifest = _manifest(self.model_dir)
        self.identity = {"repo_id": manifest["repo_id"], "revision": manifest["revision"], "max_length": max_length,
                         "template": "question:{q} \n \n passage:{p}"}
        self.name = manifest["repo_id"]
        self.requested_device = device
        self.batch_size = batch_size
        self.max_length = max_length
        self._model = None
        self._tokenizer = None
        self._device = None
        self._scores = _ArrayStore(Path(cache_dir) / "rerank-scores.sqlite3")

    def _load(self) -> None:
        import torch
        from transformers import AutoModelForSequenceClassification, AutoTokenizer

        self._device = pick_device(self.requested_device)
        self._tokenizer = AutoTokenizer.from_pretrained(str(self.model_dir), trust_remote_code=True, padding_side="left",
                                                        local_files_only=True)
        if self._tokenizer.pad_token is None:
            self._tokenizer.pad_token = self._tokenizer.eos_token
        dtype = torch.float32 if self._device == "cpu" else torch.bfloat16
        model = AutoModelForSequenceClassification.from_pretrained(str(self.model_dir), trust_remote_code=True,
                                                                   dtype=dtype, local_files_only=True).eval()
        if model.config.pad_token_id is None:
            model.config.pad_token_id = self._tokenizer.eos_token_id
        self._model = model.to(self._device)

    def score(self, query: str, passages: list[str]) -> list[float]:
        keys = [_digest({**self.identity, "query": query, "passage": passage}) for passage in passages]
        known = {key: float(value[0]) for key, value in self._scores.get(keys).items()}
        missing = [index for index, key in enumerate(keys) if key not in known]
        if missing:
            import torch

            with INFERENCE_LOCK:
                if self._model is None:
                    self._load()
                texts = [f"question:{query} \n \n passage:{passages[index]}" for index in missing]
                logits: list[float] = []
                for start in range(0, len(texts), self.batch_size):
                    batch = self._tokenizer(texts[start:start + self.batch_size], padding=True, truncation=True,
                                            return_tensors="pt", max_length=self.max_length)
                    batch = {name: tensor.to(self._device) for name, tensor in batch.items()}
                    with torch.inference_mode():
                        logits.extend(self._model(**batch).logits.view(-1).float().cpu().tolist())
            rows = [(keys[index], np.asarray([value], dtype=np.float32)) for index, value in zip(missing, logits)]
            self._scores.put(rows)
            known.update((key, float(value[0])) for key, value in rows)
        return [known[key] for key in keys]
