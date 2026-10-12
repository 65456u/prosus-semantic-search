"""Local FastAPI adapter for the shared retrieval core."""

from __future__ import annotations

import logging
import os
import threading
import time
from collections import Counter
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Literal

from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field, field_validator

from core.corpus import Corpus
from core.llm import LLM
from core.local_models import CrossEncoderReranker, DenseEncoder
from retrieval.pipeline import DENSE_INSTRUCTION, SearchPipeline
from retrieval.engine import RetrievalEngine, SemanticNotReady

ROOT = Path(__file__).resolve().parents[1]
CACHE = ROOT / ".demo-cache"
DIST = ROOT / "app" / "frontend" / "dist"
logger = logging.getLogger(__name__)
corpus = Corpus(ROOT / "data" / "5k_items_curated.csv", ROOT / "data" / "queries.csv")
category_counts = Counter(item["taxonomy_l0"] for item in corpus.items)
categories = [{"name": name, "count": count} for name, count in sorted(category_counts.items())]
local_model = CACHE / "models" / "multilingual-e5-small"
engine = RetrievalEngine(
    corpus,
    CACHE / "retrieval",
    model_path=os.environ.get("PROSUS_MODEL_PATH")
    or (str(local_model) if (local_model / "config.json").exists() else None),
    device=os.environ.get("PROSUS_DEVICE", "auto"),
    model_revision="614241f622f53c4eeff9890bdc4f31cfecc418b3",
)
_build_lock = threading.Lock()
_build_thread: threading.Thread | None = None
MODELS = CACHE / "models"
# Demo methods served by the evaluated pipeline (retrieval/pipeline.py), by pipeline system name.
PIPELINE_METHODS = {
    "harrier": ("harrier", "Harrier dense", "Local harrier-oss-v1-0.6b embeddings with an instruction-prompted query."),
    "full": ("qe-hybrid-neutral", "LLM + hybrid", "LLM query understanding (neutral prompt) with hybrid keyword and dense retrieval."),
}
_pipeline: SearchPipeline | None = None
_pipeline_lock = threading.Lock()


def pipeline_ready() -> bool:
    return all((MODELS / name / "prosus-model.json").is_file() for name in ("harrier-oss-v1-0.6b", "llama-nemotron-rerank-1b-v2"))


def get_pipeline() -> SearchPipeline:
    """Build the evaluated pipeline on first use; its models load lazily on first inference."""
    global _pipeline
    with _pipeline_lock:
        if _pipeline is None:
            if not pipeline_ready():
                raise SemanticNotReady("Download the local models first: .venv/bin/python core/download_model.py")
            _pipeline = SearchPipeline(
                engine, LLM(),
                DenseEncoder(MODELS / "harrier-oss-v1-0.6b", CACHE / "retrieval", DENSE_INSTRUCTION),
                CrossEncoderReranker(MODELS / "llama-nemotron-rerank-1b-v2", CACHE / "retrieval"),
            )
        return _pipeline


def pipeline_search(query: str, method: str, top_k: int) -> dict:
    """Run a pipeline system and shape the response like RetrievalEngine.search."""
    pipe = get_pipeline()
    system = PIPELINE_METHODS[method][0]
    start = time.perf_counter()
    understanding = pipe.understand(query, neutral=system == "qe-hybrid-neutral") if system.startswith("qe-") else None
    understood = time.perf_counter()
    result = pipe.run(system, query, top_k)
    if system == "harrier":
        dense = pipe.dense_scores(query)
        scores = {index: float(dense[index]) for index in result["ranking"]}
    else:
        by_document = result.get("rerank_scores") or result.get("fusion_scores") or {}
        scores = {index: float(by_document.get(pipe.corpus.document_ids[index], 0.0)) for index in result["ranking"]}
    finished = time.perf_counter()
    response = {
        "query": query, "method": method,
        "results": [{**corpus.items[index], "rank": rank, "score": scores[index]} for rank, index in enumerate(result["ranking"], start=1)],
        "timings_ms": {"query_encoding": (understood - start) * 1000, "retrieval": (finished - understood) * 1000,
                       "total": (finished - start) * 1000},
        "cache_hit": False, "warnings": [],
    }
    if understanding:
        response["understanding"] = {"interpretation_en": understanding["interpretation_en"], "rewrite_pt": understanding["rewrite_pt"]}
    return response


def run_method(query: str, method: str, top_k: int, use_query_cache: bool) -> dict:
    if method in PIPELINE_METHODS:
        return pipeline_search(query, method, top_k)
    return engine.search(query, method, top_k, use_query_cache)


def start_semantic_build() -> bool:
    """Keep model initialization off request threads and prevent duplicate builds."""
    global _build_thread
    with _build_lock:
        if engine.semantic_status["state"] == "ready":
            return False
        if _build_thread is not None and _build_thread.is_alive():
            return False

        def prepare() -> None:
            try:
                engine.build_semantic()
            except Exception:
                logger.exception("Semantic initialization failed")

        _build_thread = threading.Thread(target=prepare, name="e5-prepare", daemon=True)
        _build_thread.start()
        return True


@asynccontextmanager
async def lifespan(app: FastAPI):
    if os.environ.get("PROSUS_AUTO_BUILD", "1") != "0":
        start_semantic_build()
    yield


app = FastAPI(title="Prosus Search Demo", version="0.1.0", lifespan=lifespan)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://127.0.0.1:5173", "http://localhost:5173"],
    allow_methods=["GET", "POST"],
    allow_headers=["Content-Type"],
)


Method = Literal["bm25", "e5", "hybrid", "harrier", "full"]


class SearchRequest(BaseModel):
    query: str = Field(min_length=1, max_length=2000)
    method: Method = "bm25"
    top_k: int = Field(default=5, ge=1, le=20)
    use_query_cache: bool = True

    @field_validator("query")
    @classmethod
    def strip_query(cls, value: str) -> str:
        value = value.strip()
        if not value:
            raise ValueError("Enter a non-empty search query.")
        return value


class CompareRequest(BaseModel):
    query: str = Field(min_length=1, max_length=2000)
    methods: list[Method] = Field(default_factory=lambda: ["bm25", "e5", "hybrid"], min_length=1, max_length=3)
    top_k: int = Field(default=5, ge=1, le=20)
    use_query_cache: bool = True

    @field_validator("query")
    @classmethod
    def strip_query(cls, value: str) -> str:
        return SearchRequest.strip_query(value)

    @field_validator("methods")
    @classmethod
    def unique_methods(cls, values: list[Method]) -> list[Method]:
        return list(dict.fromkeys(values))


def status_payload() -> dict:
    semantic = dict(engine.semantic_status)
    return {
        "app": "prosus-search-demo",
        "pid": os.getpid(),
        "status": "ready",
        "counts": corpus.counts,
        "methods": engine.methods + [
            {"id": method, "name": name, "ready": pipeline_ready(), "description": description}
            for method, (_, name, description) in PIPELINE_METHODS.items()
        ],
        "semantic": semantic,
    }


@app.get("/api/health")
@app.get("/api/status")
def status() -> dict:
    return status_payload()


@app.post("/api/semantic/build", status_code=202)
def prepare_semantic() -> dict:
    started = start_semantic_build()
    return {"started": started, **status_payload()}


@app.get("/api/queries")
def queries() -> dict:
    return {"queries": corpus.queries}


@app.get("/api/catalog")
def catalog(
    query: str = Query(default="", max_length=2000),
    category: str = Query(default="", max_length=200),
    offset: int = Query(default=0, ge=0),
    limit: int = Query(default=24, ge=1, le=100),
) -> dict:
    result = corpus.catalog(query=query, category=category, offset=offset, limit=limit)
    result["categories"] = categories
    return result


@app.get("/api/items/{document_id}")
def item(document_id: str) -> dict:
    try:
        return corpus.get_item(document_id)
    except KeyError as exc:
        raise HTTPException(status_code=404, detail="This item was not found.") from exc


@app.post("/api/search")
def search(request: SearchRequest) -> dict:
    try:
        return run_method(request.query, request.method, request.top_k, request.use_query_cache)
    except SemanticNotReady as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    except RuntimeError as exc:  # local LLM server unreachable for a query outside the cache
        logger.exception("Search failed for method %s", request.method)
        raise HTTPException(status_code=503, detail=f"The local LLM is unavailable ({exc}). Start it with `ollama serve`.") from exc
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    except Exception as exc:
        logger.exception("Search failed for method %s", request.method)
        raise HTTPException(status_code=500, detail="Search failed. Check the local server log.") from exc


@app.post("/api/compare")
def compare(request: CompareRequest) -> dict:
    # Serial calls keep the demo's displayed stage timings free of self-created
    # parallel model contention. A dedicated load test can exercise concurrency.
    runs = []
    for method in request.methods:
        try:
            runs.append(run_method(request.query, method, request.top_k, request.use_query_cache))
        except SemanticNotReady as exc:
            runs.append({"method": method, "error": str(exc)})
        except Exception:
            logger.exception("Compare failed for method %s", method)
            runs.append({"method": method, "error": "Search failed. Check the local server log."})
    return {"query": request.query, "runs": runs}


if (DIST / "assets").is_dir():
    app.mount("/assets", StaticFiles(directory=DIST / "assets"), name="assets")


@app.get("/")
def frontend():
    if not (DIST / "index.html").is_file():
        raise HTTPException(status_code=503, detail="Build the frontend first, or open the Vite development server.")
    return FileResponse(DIST / "index.html")


@app.get("/favicon.svg")
def favicon():
    path = DIST / "favicon.svg"
    if path.is_file():
        return FileResponse(path)
    raise HTTPException(status_code=404)
