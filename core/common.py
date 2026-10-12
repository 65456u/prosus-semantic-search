"""Shared loading of the corpus, retrieval engine, local models, and LLM pipeline."""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any, Iterable

from core.corpus import Corpus
from core.llm import LLM
from core.local_models import CrossEncoderReranker, DenseEncoder
from retrieval.pipeline import DENSE_INSTRUCTION, SearchPipeline
from retrieval.engine import RetrievalEngine

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "output" / "eval"
RUNS = OUT / "runs.jsonl"
QUERY_INFO = OUT / "queries.jsonl"
JUDGMENTS = OUT / "judgments.jsonl"
HUMAN = OUT / "human_labels.jsonl"
POOL_DEPTH = 5


def load_corpus() -> Corpus:
    return Corpus(ROOT / "data" / "5k_items_curated.csv", ROOT / "data" / "queries.csv")


def load_pipeline(corpus: Corpus | None = None, offline: bool = False) -> SearchPipeline:
    corpus = corpus or load_corpus()
    engine = RetrievalEngine(corpus, ROOT / ".demo-cache/retrieval", ROOT / ".demo-cache/models/multilingual-e5-small",
                             device=os.environ.get("PROSUS_DEVICE", "auto"), model_revision="614241f622f53c4eeff9890bdc4f31cfecc418b3")
    engine.build_semantic()
    models = ROOT / ".demo-cache/models"
    dense = DenseEncoder(models / "harrier-oss-v1-0.6b", ROOT / ".demo-cache/retrieval", DENSE_INSTRUCTION)
    reranker = CrossEncoderReranker(models / "llama-nemotron-rerank-1b-v2", ROOT / ".demo-cache/retrieval")
    return SearchPipeline(engine, LLM(offline=offline), dense, reranker)


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.is_file():
        return []
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def write_jsonl(path: Path, records: Iterable[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as stream:
        for record in records:
            stream.write(json.dumps(record, ensure_ascii=False) + "\n")
    temporary.replace(path)
