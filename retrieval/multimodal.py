"""Industry-style baseline: one LLM-based multimodal embedder (Qwen3-VL-Embedding-2B) for text and photos.

    .venv/bin/python core/download_model.py --only qwen3-vl-embedding-2b
    .venv/bin/python -m core.images                 # every listing photo (~75 MB)
    .venv/bin/python -m retrieval.multimodal [--batch-size 8]

Production food search (Uber Eats, DoorDash, Meituan) has moved to LLM-backbone two-tower
embedders that also read the item photo. Qwen3-VL-Embedding-2B is the open model of that kind
that still fits next to the demo on a 16 GB Mac. It is used zero-shot. Systems:

    vl-text    query vs listing text (the same text harrier reads)
    vl-image   query vs listing photo only; listings without a photo fall back to their text vector
    vl-fused   query vs listing text and photo in one input
    hybrid-vl  RRF(bm25-pt, harrier, vl-fused): the hybrid plus a multimodal channel
    qe-hybrid-vl  qe-hybrid-neutral plus vl-fused on the raw query and on the LLM expansion:
                  are query understanding and photos complementary?

Queries carry the same English task instruction as harrier; documents use the model's default
instruction. Vectors are cached per model revision and exact input (photo by content hash).
Runs on the 100 evaluation queries are added to runs.jsonl; judge them with evaluation.judge.
The VL systems are also scored on held-out historical search terms (behavioral.json), an
LLM-free check that matters here because the judge shares Qwen3-VL's model family.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import time
from pathlib import Path

import numpy as np

from core.local_models import _ArrayStore, _digest, _manifest, pick_device
from retrieval.pipeline import CANDIDATE_DEPTH, DENSE_INSTRUCTION

from evaluation.behavioral import RESULT as BEHAVIOURAL, evaluate_terms, held_out_terms, paired
from core.common import ROOT, RUNS, load_pipeline, read_jsonl, write_jsonl
from core.images import image_path

MODEL_DIR = ROOT / ".demo-cache" / "models" / "qwen3-vl-embedding-2b"
VL_SYSTEMS = ["vl-text", "vl-image", "vl-fused", "hybrid-vl", "qe-hybrid-vl"]
CHANNELS = ("text", "image", "fused")


class VLEncoder:
    def __init__(self, model_dir: Path = MODEL_DIR, cache_dir: Path = ROOT / ".demo-cache" / "retrieval",
                 device: str = "auto", batch_size: int = 8, max_seq_length: int = 512):
        manifest = _manifest(model_dir)
        self.model_dir, self.batch_size, self.device = model_dir, batch_size, pick_device(device)
        # fp16 on the GPU: closer to fp32 than bf16 here (min cosine 0.9995 vs 0.997) and 2.5x faster on MPS.
        self.dtype = "float32" if self.device == "cpu" else "float16"
        self.identity = {"repo_id": manifest["repo_id"], "revision": manifest["revision"], "max_seq_length": max_seq_length,
                         "dtype": self.dtype}
        self.store = _ArrayStore(Path(cache_dir) / "vl-vectors.sqlite3")
        self.max_seq_length = max_seq_length
        self._model = None

    @property
    def model(self):
        if self._model is None:
            import torch
            from sentence_transformers import SentenceTransformer

            self._model = SentenceTransformer(str(self.model_dir), device=self.device, local_files_only=True,
                                              model_kwargs={"dtype": getattr(torch, self.dtype)})
            self._model.max_seq_length = self.max_seq_length
        return self._model

    def encode(self, inputs: list, keys: list[dict], prompt: str | None = None, label: str = "",
               prepare=lambda value: value) -> np.ndarray:
        """Vectors for inputs, cached by `keys` (JSON-able identities). `prepare` turns an input into what the
        model takes (str, PIL image, or {"text", "image"}) just before encoding, so photos load one chunk at a time."""
        digests = [_digest({**self.identity, "prompt": prompt, **key}) for key in keys]
        known = self.store.get(digests)
        missing = [i for i, d in enumerate(digests) if d not in known]
        start = time.perf_counter()
        for offset in range(0, len(missing), self.batch_size * 16):
            chunk = missing[offset:offset + self.batch_size * 16]
            vectors = self.model.encode([prepare(inputs[i]) for i in chunk], prompt=prompt, batch_size=self.batch_size,
                                        normalize_embeddings=True, convert_to_numpy=True)
            rows = [(digests[i], vector.astype(np.float32)) for i, vector in zip(chunk, vectors)]
            self.store.put(rows)
            known.update(rows)
            if label:
                done = offset + len(chunk)
                print(f"  {label}: {done}/{len(missing)} encoded ({time.perf_counter() - start:.0f}s)", flush=True)
        return np.stack([known[d] for d in digests])

    def queries(self, texts: list[str]) -> np.ndarray:
        return self.encode(texts, [{"query": text} for text in texts], prompt=DENSE_INSTRUCTION)


def photo_hash(path: Path) -> str:
    return hashlib.sha1(path.read_bytes()).hexdigest()


MAX_PIXELS = 300 * 300  # the CDN's t_low size; a few listings link larger originals (up to 1.5 MP)


def load_photo(path: Path):
    """RGB photo scaled down to at most MAX_PIXELS, so every photo costs about 90 image tokens."""
    from PIL import Image

    image = Image.open(path).convert("RGB")
    width, height = image.size
    if width * height > MAX_PIXELS:
        scale = (MAX_PIXELS / (width * height)) ** 0.5
        image = image.resize((max(1, int(width * scale)), max(1, int(height * scale))), Image.Resampling.BICUBIC)
    return image


def document_matrices(encoder: VLEncoder, corpus) -> dict[str, np.ndarray]:
    texts = corpus.texts
    photos = []
    for item in corpus.items:
        path = image_path(item["image_url"]) if item["image_url"] else None
        photos.append(path if path and path.is_file() and path.stat().st_size else None)
    has_photo = [i for i, path in enumerate(photos) if path]
    print(f"{len(has_photo)}/{len(texts)} listings have a downloaded photo", flush=True)
    text = encoder.encode(texts, [{"text": t} for t in texts], label="text")
    image = text.copy()
    if has_photo:
        image[has_photo] = encoder.encode([photos[i] for i in has_photo],
                                          [{"image": photo_hash(photos[i]), "max_pixels": MAX_PIXELS} for i in has_photo],
                                          label="image", prepare=load_photo)
    fused = text.copy()
    if has_photo:
        fused[has_photo] = encoder.encode([(texts[i], photos[i]) for i in has_photo],
                                          [{"text": texts[i], "image": photo_hash(photos[i]), "max_pixels": MAX_PIXELS}
                                           for i in has_photo],
                                          label="fused", prepare=lambda pair: {"text": pair[0], "image": load_photo(pair[1])})
    return {"text": text, "image": image, "fused": fused}


def hybrid_vl(pipeline, query: str, vector: np.ndarray, fused: np.ndarray) -> list[int]:
    """RRF over stemmed BM25, harrier, and the Qwen3-VL text+photo channel."""
    engine = pipeline.engine
    ranking, _ = pipeline._fuse([
        engine.rank_distinct(engine.bm25_scores(query, stem=True), CANDIDATE_DEPTH, positive_only=True),
        engine.rank_distinct(pipeline.dense_scores(query), CANDIDATE_DEPTH),
        engine.rank_distinct(fused @ vector, CANDIDATE_DEPTH)])
    return ranking


def retrievers(pipeline, encoder: VLEncoder | None = None) -> dict:
    """system -> function(query) -> top-10 row indices for the photo-aware systems (document vectors are cached)."""
    engine = pipeline.engine
    encoder = encoder or VLEncoder()
    matrices = document_matrices(encoder, pipeline.corpus)
    _ = pipeline.dense_matrix

    def vector(text: str) -> np.ndarray:
        return encoder.queries([text])[0]

    def qe_vl(query: str) -> list[int]:
        understanding = pipeline.understand(query, neutral=True)
        ranking, _ = pipeline._fuse(pipeline.expanded_rankings(query, understanding) + [
            engine.rank_distinct(matrices["fused"] @ vector(query), CANDIDATE_DEPTH),
            engine.rank_distinct(matrices["fused"] @ vector(pipeline.dense_expansion(understanding)), CANDIDATE_DEPTH)])
        return ranking[:10]

    return {"vl-text": lambda q: engine.rank_distinct(matrices["text"] @ vector(q), 10),
            "vl-image": lambda q: engine.rank_distinct(matrices["image"] @ vector(q), 10),
            "vl-fused": lambda q: engine.rank_distinct(matrices["fused"] @ vector(q), 10),
            "hybrid-vl": lambda q: hybrid_vl(pipeline, q, vector(q), matrices["fused"])[:10],
            "qe-hybrid-vl": qe_vl, "_encoder": encoder}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--top-k", type=int, default=10)
    args = parser.parse_args()
    pipeline = load_pipeline()
    corpus, engine = pipeline.corpus, pipeline.engine
    encoder = VLEncoder(batch_size=args.batch_size)
    start = time.perf_counter()
    matrices = document_matrices(encoder, corpus)
    print(f"Document vectors ready ({time.perf_counter() - start:.0f}s)", flush=True)
    query_vectors = encoder.queries([query["text"] for query in corpus.queries])
    # Query understanding with the neutral prompt (cached from the qe-hybrid-neutral runs).
    understandings = [pipeline.understand(query["text"], neutral=True) for query in corpus.queries]
    expansion_vectors = encoder.queries([pipeline.dense_expansion(u) for u in understandings])
    _ = pipeline.dense_matrix
    records = []
    for system in VL_SYSTEMS:
        for query, vector, understanding, expansion in zip(corpus.queries, query_vectors, understandings, expansion_vectors):
            start = time.perf_counter()
            if system == "qe-hybrid-vl":
                ranking, _ = pipeline._fuse(pipeline.expanded_rankings(query["text"], understanding) + [
                    engine.rank_distinct(matrices["fused"] @ vector, CANDIDATE_DEPTH),
                    engine.rank_distinct(matrices["fused"] @ expansion, CANDIDATE_DEPTH)])
            elif system == "hybrid-vl":
                ranking = hybrid_vl(pipeline, query["text"], vector, matrices["fused"])
            else:
                ranking = engine.rank_distinct(matrices[system.removeprefix("vl-")] @ vector, args.top_k)
            records.append({"query_id": query["id"], "query": query["text"], "system": system,
                            "document_ids": [corpus.document_ids[i] for i in ranking[:args.top_k]],
                            "elapsed_ms": round((time.perf_counter() - start) * 1000, 1)})
    write_jsonl(RUNS, [r for r in read_jsonl(RUNS) if r["system"] not in VL_SYSTEMS] + records)
    print(f"Saved {len(records)} runs to {RUNS}")

    # LLM-free check on held-out historical search terms. The judge shares Qwen3-VL's model family,
    # so a gain that also shows up here is not judge self-preference.
    terms = held_out_terms(corpus)
    names = sorted(terms)
    term_vectors = dict(zip(names, encoder.queries(names)))
    result = json.loads(BEHAVIOURAL.read_text()) if BEHAVIOURAL.is_file() else {"systems": {}}
    retrieve = {
        "vl-text": lambda term: engine.rank_distinct(matrices["text"] @ term_vectors[term], 10),
        "vl-image": lambda term: engine.rank_distinct(matrices["image"] @ term_vectors[term], 10),
        "vl-fused": lambda term: engine.rank_distinct(matrices["fused"] @ term_vectors[term], 10),
        "hybrid-vl": lambda term: hybrid_vl(pipeline, term, term_vectors[term], matrices["fused"]),
    }
    comparisons = {}
    for system, function in retrieve.items():
        result["systems"][system] = evaluate_terms(corpus, terms, names, function)
        baseline = {"vl-text": "harrier", "vl-fused": "harrier", "hybrid-vl": "hybrid"}.get(system)
        if baseline and (comparison := paired(result, system, baseline)):
            comparisons[system] = comparison
        print(f"{system:10} held-out terms {len(names)}  MRR@10 {result['systems'][system]['mrr@10']:.3f}", flush=True)
    result["vl_comparisons"] = comparisons
    BEHAVIOURAL.write_text(json.dumps(result, indent=1, ensure_ascii=False), encoding="utf-8")
    print(f"Saved {BEHAVIOURAL}")


if __name__ == "__main__":
    main()
