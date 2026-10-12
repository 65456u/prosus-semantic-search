"""Reusable local BM25, multilingual E5, and reciprocal-rank fusion baselines.

Scores are ranking scores, not relevance probabilities. The corpus, document
vectors, and query vectors are shared across the API and offline experiments.
No paid API is called by this module.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import sqlite3
import threading
import time
from pathlib import Path
from typing import Any

import numpy as np
import snowballstemmer
from rank_bm25 import BM25Okapi

from core.corpus import Corpus, normalize_text
from core.local_models import INFERENCE_LOCK


MODEL_ID = "intfloat/multilingual-e5-small"
ENCODING_CONFIG = {
    "representation": "product-metadata-pt-v1",
    "max_seq_length": 512,
    "normalize_embeddings": True,
    "dtype": "float32",
    "query_prefix": "query: ",
    "document_prefix": "passage: ",
}
_STOPWORDS = frozenset("a as o os de da das do dos e em no nos na nas um uma uns umas ao aos à às por para que com se é".split())


class SemanticNotReady(RuntimeError):
    """E5 or hybrid was requested before document encoding completed."""


_STEMMER = snowballstemmer.stemmer("portuguese")
_STEM_LOCK = threading.Lock()


def tokenize_portuguese(text: str, stem: bool = False) -> list[str]:
    """Deterministic accent-insensitive BM25 tokens with a small stopword list.

Negation such as 'sem' is retained. Without `stem` there is no stemming. With
`stem`, the Snowball Portuguese stemmer runs on the accented word before accent
removal, so 'batatas fritas' and 'batata frita' share tokens.
"""
    if not stem:
        return [token for token in re.findall(r"[a-z0-9]+", normalize_text(text)) if token not in _STOPWORDS]
    words = [word for word in re.findall(r"[^\W_]+", str(text).casefold()) if normalize_text(word) not in _STOPWORDS]
    with _STEM_LOCK:  # the Snowball stemmer object is not thread-safe
        stems = _STEMMER.stemWords(words)
    return [normalize_text(token) for token in stems]


def reciprocal_rank_fusion(rankings: list[list[int]], constant: int = 60) -> dict[int, float]:
    """Canonical RRF: sum 1/(constant + one-based rank) for each candidate."""
    if constant < 0:
        raise ValueError("RRF constant must be non-negative.")
    scores: dict[int, float] = {}
    for ranking in rankings:
        seen = set()
        for rank, document_index in enumerate(ranking, start=1):
            if document_index in seen:
                continue
            seen.add(document_index)
            scores[document_index] = scores.get(document_index, 0.0) + 1.0 / (constant + rank)
    return scores


def _digest(value: Any) -> str:
    return hashlib.sha256(json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def _normalize_vectors(vectors: Any, expected_rows: int) -> np.ndarray:
    vectors = np.asarray(vectors, dtype=np.float32)
    if vectors.ndim != 2 or vectors.shape[0] != expected_rows or not np.isfinite(vectors).all():
        raise ValueError("Encoder returned invalid embedding dimensions or non-finite values.")
    norms = np.linalg.norm(vectors, axis=1, keepdims=True)
    if np.any(norms <= 0):
        raise ValueError("Encoder returned a zero-length vector.")
    return vectors / norms


class RetrievalEngine:
    def __init__(self, corpus: Corpus, cache_dir: str | Path, model_path: str | Path | None = None,
                 device: str = "auto", model_revision: str | None = None):
        self.corpus = corpus
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.model_path = Path(model_path) if model_path else None
        self.requested_device = device
        self.model_revision = model_revision
        self._status_lock = threading.Lock()
        self._build_lock = threading.Lock()
        self._encode_lock = INFERENCE_LOCK  # shared with every local model; see local_models.py
        self._status = {"state": "not_started", "progress": None, "message": "E5 has not started loading.", "model": MODEL_ID, "device": None}
        self._encoder = None
        self._model_identity = None
        self._device = None
        self._embeddings: np.ndarray | None = None
        self._bm25 = BM25Okapi([tokenize_portuguese(text) for text in corpus.texts], k1=1.5, b=0.75)
        self._bm25_stemmed = BM25Okapi([tokenize_portuguese(text, stem=True) for text in corpus.texts], k1=1.5, b=0.75)
        self._initialize_vector_cache(self.cache_dir / "docs.sqlite3")
        self._initialize_vector_cache(self.cache_dir / "queries.sqlite3")

    @property
    def semantic_status(self) -> dict[str, Any]:
        with self._status_lock:
            return self._status.copy()

    @property
    def methods(self) -> list[dict[str, Any]]:
        ready = self.semantic_status["state"] == "ready"
        return [
            {"id": "bm25", "name": "BM25", "ready": True, "description": "Portuguese keyword matching; accent-insensitive tokens."},
            {"id": "e5", "name": "Multilingual E5", "ready": ready, "description": "Local multilingual-e5-small embeddings with exact cosine search."},
            {"id": "hybrid", "name": "Hybrid · BM25 + E5", "ready": ready, "description": "Reciprocal-rank fusion of the top 100 distinct listings from each retriever."},
        ]

    def _set_status(self, **updates: Any) -> None:
        with self._status_lock:
            self._status.update(updates)

    @staticmethod
    def _initialize_vector_cache(path: Path) -> None:
        with sqlite3.connect(path, timeout=30) as connection:
            connection.execute("PRAGMA journal_mode=WAL")
            connection.execute("CREATE TABLE IF NOT EXISTS vectors (key TEXT PRIMARY KEY, dims INTEGER NOT NULL, vector BLOB NOT NULL)")

    @staticmethod
    def _read_vectors(path: Path, keys: list[str]) -> dict[str, np.ndarray]:
        found = {}
        with sqlite3.connect(path, timeout=30) as connection:
            # Small batches avoid SQLite's parameter limits on larger corpora.
            for start in range(0, len(keys), 400):
                batch = keys[start:start + 400]
                if not batch:
                    continue
                rows = connection.execute(f"SELECT key,dims,vector FROM vectors WHERE key IN ({','.join('?' for _ in batch)})", batch)
                for key, dims, blob in rows:
                    vector = np.frombuffer(blob, dtype=np.float32).copy()
                    if vector.shape == (dims,) and np.isfinite(vector).all() and np.linalg.norm(vector) > 0:
                        found[key] = vector
        return found

    @staticmethod
    def _write_vectors(path: Path, rows: list[tuple[str, np.ndarray]]) -> None:
        with sqlite3.connect(path, timeout=30) as connection:
            connection.executemany("INSERT OR REPLACE INTO vectors(key,dims,vector) VALUES (?,?,?)", [
                (key, int(vector.shape[0]), np.asarray(vector, dtype=np.float32).tobytes()) for key, vector in rows
            ])

    def _vector_key(self, input_text: str) -> str:
        return _digest({"model_identity": self._model_identity, "encoding": ENCODING_CONFIG, "input": input_text})

    def _load_model(self) -> tuple[Any, str, str]:
        """Resolve an exact local model snapshot, then load it once.

        A file digest supplements the pinned revision: edits to a local model
        directory invalidate caches even when a manifest is unchanged. Tests
        can replace this method with a deterministic encoder and identity.
        """
        import torch
        from sentence_transformers import SentenceTransformer

        if self.requested_device == "auto":
            device = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
        else:
            device = self.requested_device
        # Large BLAS/thread pools hurt interactive latency on a 5k-row corpus.
        torch.set_num_threads(max(1, min(4, os.cpu_count() or 1)))
        model_path = self.model_path
        if model_path is None:
            from huggingface_hub import snapshot_download
            self._set_status(message="Downloading the open-source E5 model.", device=device)
            model_path = Path(snapshot_download(
                repo_id=MODEL_ID, revision=self.model_revision,
                cache_dir=str(self.cache_dir / "models"),
                allow_patterns=["*.json", "*.txt", "*.safetensors", "1_Pooling/*"],
            ))
        if not model_path.is_dir():
            raise FileNotFoundError(f"Model directory does not exist: {model_path}")
        manifest_path = model_path / "prosus-model.json"
        manifest = json.loads(manifest_path.read_text()) if manifest_path.exists() else {}
        file_hash = hashlib.sha256()
        model_files = sorted(path for path in model_path.rglob("*") if path.is_file() and path.suffix in (".json", ".txt", ".safetensors", ".bin") and not any(part in ("onnx", "openvino", ".cache") for part in path.relative_to(model_path).parts))
        for path in model_files:
            file_hash.update(str(path.relative_to(model_path)).encode())
            with path.open("rb") as handle:
                while chunk := handle.read(1024 * 1024):
                    file_hash.update(chunk)
        if not model_files:
            raise ValueError("The model directory contains no model files.")
        revision = manifest.get("revision") or self.model_revision or (model_path.name if model_path.parent.name == "snapshots" else "local-files")
        identity = _digest({"repo_id": manifest.get("repo_id", MODEL_ID), "revision": revision, "files_sha256": file_hash.hexdigest()})
        self._set_status(message="Loading E5 for local inference.", device=device, revision=revision)
        model = SentenceTransformer(str(model_path), device=device, local_files_only=True)
        model.max_seq_length = ENCODING_CONFIG["max_seq_length"]
        return model, identity, str(model.device)

    def build_semantic(self) -> None:
        """Synchronous initialization; call from a background worker.

        Progress is a fraction in [0, 1]. Embeddings are persisted both per
        input and as a complete matrix. A changed model, representation,
        encoding setting, or source input cannot reuse stale vectors.
        """
        if not self._build_lock.acquire(blocking=False):
            return
        try:
            if self._embeddings is not None:
                return
            self._set_status(state="loading", progress=0.0, message="Preparing multilingual E5.")
            encoder, identity, device = self._load_model()
            self._encoder, self._model_identity, self._device = encoder, identity, device
            inputs = [ENCODING_CONFIG["document_prefix"] + text for text in self.corpus.texts]
            keys = [self._vector_key(text) for text in inputs]
            digest = _digest({"model_identity": identity, "encoding": ENCODING_CONFIG, "documents": list(zip(self.corpus.document_ids, keys))})
            matrix_dir = self.cache_dir / "doc-embeddings"
            matrix_dir.mkdir(exist_ok=True)
            matrix_path, metadata_path = matrix_dir / f"{digest}.npy", matrix_dir / f"{digest}.json"
            if matrix_path.exists() and metadata_path.exists():
                metadata = json.loads(metadata_path.read_text())
                vectors = np.load(matrix_path, allow_pickle=False)
                if metadata.get("key") == digest and metadata.get("model_identity") == identity and metadata.get("document_ids") == self.corpus.document_ids and vectors.shape == tuple(metadata.get("shape", [])) and vectors.shape[0] == len(inputs):
                    self._embeddings = _normalize_vectors(vectors, len(inputs))
                    self._set_status(state="ready", progress=1.0, message="E5 is ready. Document embeddings loaded from disk.", device=device, document_cache_hit=True)
                    return
            known = self._read_vectors(self.cache_dir / "docs.sqlite3", keys)
            # Encode each distinct input once, preserving every source row.
            missing = list(dict.fromkeys(key for key in keys if key not in known))
            missing_set = set(missing)
            inputs_by_key = dict(zip(keys, inputs))
            completed = len(keys) - sum(key in missing_set for key in keys)
            self._set_status(progress=completed / max(len(keys), 1), message=f"Encoding {len(missing):,} product descriptions.", device=device)
            for start in range(0, len(missing), 32):
                batch_keys = missing[start:start + 32]
                with self._encode_lock:
                    encoded = encoder.encode([inputs_by_key[key] for key in batch_keys], batch_size=32, show_progress_bar=False, convert_to_numpy=True, normalize_embeddings=True)
                encoded = _normalize_vectors(encoded, len(batch_keys))
                rows = list(zip(batch_keys, encoded))
                self._write_vectors(self.cache_dir / "docs.sqlite3", rows)
                known.update(rows)
                progress = min(start + len(batch_keys), len(missing)) / max(len(missing), 1)
                fraction = (completed + (len(keys) - completed) * progress) / max(len(keys), 1)
                self._set_status(progress=fraction, message=f"Encoding products · {int(fraction * 100)}% complete.")
            vectors = _normalize_vectors(np.stack([known[key] for key in keys]), len(inputs))
            temp_matrix = matrix_path.with_suffix(".npy.tmp")
            with temp_matrix.open("wb") as handle:
                np.save(handle, vectors, allow_pickle=False)
            os.replace(temp_matrix, matrix_path)
            metadata = {"key": digest, "model": MODEL_ID, "model_identity": identity, "encoding": ENCODING_CONFIG, "document_ids": self.corpus.document_ids, "input_keys": keys, "shape": list(vectors.shape)}
            temp_metadata = metadata_path.with_suffix(".json.tmp")
            temp_metadata.write_text(json.dumps(metadata, ensure_ascii=False), encoding="utf-8")
            os.replace(temp_metadata, metadata_path)
            self._embeddings = vectors
            self._set_status(state="ready", progress=1.0, message="E5 is ready. All document embeddings are cached.", device=device, document_cache_hit=not missing)
        except Exception as exc:
            self._encoder = None
            self._embeddings = None
            self._set_status(state="error", progress=None, message=f"E5 initialization failed: {type(exc).__name__}: {exc}")
            raise
        finally:
            self._build_lock.release()

    def _query_vector(self, query: str, use_cache: bool) -> tuple[np.ndarray, bool]:
        input_text = ENCODING_CONFIG["query_prefix"] + query.strip()
        key = self._vector_key(input_text)
        if use_cache:
            known = self._read_vectors(self.cache_dir / "queries.sqlite3", [key])
            if key in known and known[key].shape == (self._embeddings.shape[1],):
                return known[key], True
        with self._encode_lock:
            # A second read avoids duplicate computation during concurrent requests.
            if use_cache:
                known = self._read_vectors(self.cache_dir / "queries.sqlite3", [key])
                if key in known and known[key].shape == (self._embeddings.shape[1],):
                    return known[key], True
            encoded = self._encoder.encode([input_text], batch_size=32, show_progress_bar=False, convert_to_numpy=True, normalize_embeddings=True)
            vector = _normalize_vectors(encoded, 1)[0]
            if vector.shape != (self._embeddings.shape[1],):
                raise ValueError("Query and document embedding dimensions do not match.")
            if use_cache:
                self._write_vectors(self.cache_dir / "queries.sqlite3", [(key, vector)])
            return vector, False

    def _rank_distinct(self, scores: np.ndarray, limit: int, positive_only: bool = False) -> list[int]:
        # Stable sort uses source row order as the deterministic tie-breaker.
        order = np.argsort(-scores, kind="stable")
        ranked, seen = [], set()
        for index in order:
            if positive_only and scores[index] <= 0:
                break
            item = self.corpus.items[int(index)]
            listing = (item["item_id"], item["merchant_id"])
            if listing in seen:
                continue
            seen.add(listing)
            ranked.append(int(index))
            if len(ranked) >= limit:
                break
        return ranked

    def bm25_scores(self, query: str, stem: bool = False) -> np.ndarray:
        index = self._bm25_stemmed if stem else self._bm25
        return np.asarray(index.get_scores(tokenize_portuguese(query, stem=stem)), dtype=np.float64)

    def e5_scores(self, query: str, use_cache: bool = True) -> np.ndarray:
        if self.semantic_status["state"] != "ready":
            raise SemanticNotReady(self.semantic_status["message"])
        query_vector, _ = self._query_vector(query, use_cache)
        return self._embeddings @ query_vector

    def rank_distinct(self, scores: np.ndarray, limit: int, positive_only: bool = False) -> list[int]:
        """Public form of distinct-listing ranking for pipelines built on this engine."""
        return self._rank_distinct(scores, limit, positive_only)

    def search(self, query: str, method: str, top_k: int = 5, use_query_cache: bool = True) -> dict[str, Any]:
        """Search with live timing and optional persistent query-vector reuse.

        query_encoding includes query-vector cache lookup and any model wait;
        retrieval includes scoring, sorting, deduplication, and result assembly.
        total excludes HTTP transport and client rendering. cache_hit describes
        query-vector reuse only: final search results are always scored live.
        Disabling query caching disables both reads and writes for benchmarks.
        """
        if not isinstance(query, str) or not query.strip():
            raise ValueError("Query must contain text.")
        if method not in {"bm25", "e5", "hybrid"}:
            raise ValueError(f"Unknown retrieval method: {method}")
        if not 1 <= top_k <= 100:
            raise ValueError("top_k must be between 1 and 100.")
        if method != "bm25" and self.semantic_status["state"] != "ready":
            raise SemanticNotReady(self.semantic_status["message"])
        start = time.perf_counter()
        encoding_ms, cache_hit = 0.0, False
        if method != "bm25":
            encoding_start = time.perf_counter()
            query_vector, cache_hit = self._query_vector(query, use_query_cache)
            encoding_ms = (time.perf_counter() - encoding_start) * 1000
        retrieval_start = time.perf_counter()
        warnings = []
        if method == "bm25":
            scores = self._bm25.get_scores(tokenize_portuguese(query))
            supported = self._rank_distinct(scores, top_k, positive_only=True)
            ranking = self._rank_distinct(scores, top_k)
            if len(supported) < top_k:
                warnings.append(
                    f"BM25 found {len(supported)} distinct listings with a positive keyword score. "
                    "Remaining zero-score items are tied in source-row order and are not supported lexical matches."
                )
        elif method == "e5":
            scores = self._embeddings @ query_vector
            ranking = self._rank_distinct(scores, top_k)
        else:
            lexical_scores = self._bm25.get_scores(tokenize_portuguese(query))
            dense_scores = self._embeddings @ query_vector
            lexical = self._rank_distinct(lexical_scores, 100, positive_only=True)
            dense = self._rank_distinct(dense_scores, 100)
            # Fuse by distinct listing. Duplicate source rows can rank differently
            # in the two branches but must represent the same candidate in RRF.
            representative = {}
            def canonical(indices: list[int]) -> list[int]:
                canonical_indices = []
                for index in indices:
                    item = self.corpus.items[index]
                    listing = (item["item_id"], item["merchant_id"])
                    canonical_indices.append(representative.setdefault(listing, index))
                return canonical_indices
            fused = reciprocal_rank_fusion([canonical(lexical), canonical(dense)])
            ranking = sorted(fused, key=lambda index: (-fused[index], index))[:top_k]
            scores = fused
        results = [{**self.corpus.items[index], "rank": rank, "score": float(scores[index])} for rank, index in enumerate(ranking, start=1)]
        return {
            "query": query,
            "method": method,
            "results": results,
            "timings_ms": {"query_encoding": encoding_ms, "retrieval": (time.perf_counter() - retrieval_start) * 1000, "total": (time.perf_counter() - start) * 1000},
            "cache_hit": cache_hit,
            "warnings": warnings,
        }
