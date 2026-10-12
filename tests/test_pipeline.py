"""Offline tests for the ablation pipeline, the LLM cache, and local-model caches (no model downloads)."""

from __future__ import annotations

import json

import numpy as np
import pytest

from core.corpus import Corpus
from core.llm import LLM, split_model
from core.local_models import DenseEncoder
from retrieval.pipeline import SYSTEMS, SearchPipeline
from retrieval.engine import RetrievalEngine, tokenize_portuguese

from conftest import DummyEncoder


class FakeDense:
    def __init__(self):
        self.encoder = DummyEncoder("dense")

    def documents(self, texts):
        return self.encoder.encode(texts)

    def queries(self, texts):
        return self.encoder.encode(["query " + text for text in texts])


class FakeReranker:
    """Prefers passages that mention 'Sorvete', so reranking has a visible effect."""

    def score(self, query, passages):
        return [5.0 if "Sorvete" in passage else -1.0 for passage in passages]


class FakeLLM:
    def __init__(self):
        self.calls = 0

    def json(self, task, model, system, user, schema, **kwargs):
        self.calls += 1
        return {"interpretation_en": "cheese bread", "rewrite_pt": "pão de queijo mineiro",
                "dish_names_pt": ["pão de queijo"], "ingredients_pt": ["queijo"]}


@pytest.fixture
def pipeline(tiny_csvs, tmp_path):
    corpus = Corpus(*tiny_csvs)
    engine = RetrievalEngine(corpus, tmp_path / "cache")
    encoder = DummyEncoder()
    engine._load_model = lambda: (encoder, encoder.variant, "cpu")
    engine.build_semantic()
    return SearchPipeline(engine, FakeLLM(), FakeDense(), FakeReranker())


def test_stemmed_tokens_match_portuguese_inflections():
    assert tokenize_portuguese("Batatas FRITAS", stem=True) == tokenize_portuguese("batata frita", stem=True)
    assert tokenize_portuguese("Batatas fritas") != tokenize_portuguese("batata frita")
    assert "sem" in tokenize_portuguese("Pão sem glúten", stem=True)


@pytest.mark.parametrize("system", SYSTEMS)
def test_every_system_returns_distinct_valid_listings(pipeline, system):
    result = pipeline.run(system, "pães de queijo", top_k=3)
    ranking = result["ranking"]
    assert 1 <= len(ranking) <= 3
    assert len(set(ranking)) == len(ranking)
    assert all(0 <= index < len(pipeline.corpus.items) for index in ranking)


def test_reranking_reorders_candidates_and_reports_scores(pipeline):
    reranked = pipeline.run("qe-hybrid-rerank", "pão de queijo", top_k=4)
    assert pipeline.corpus.items[reranked["ranking"][0]]["name"] == "Sorvete de morango"
    assert reranked["max_rerank_score"] == 5.0
    assert reranked["understanding"]["rewrite_pt"] == "pão de queijo mineiro"
    with pytest.raises(ValueError):
        pipeline.run("unknown", "pizza")


def test_llm_cache_ledger_and_model_names(tmp_path, monkeypatch):
    llm = LLM(cache_dir=tmp_path)
    calls = []

    def fake_ollama(task, request):
        calls.append(request)
        return {"grade": 2}, 100, 5

    monkeypatch.setattr(llm, "_ollama", fake_ollama)
    schema = {"type": "object", "properties": {"grade": {"type": "integer"}}}
    first = llm.json("judge", "ollama/qwen3.5:9b-q4_K_M", "system", "user", schema)
    second = llm.json("judge", "ollama/qwen3.5:9b-q4_K_M", "system", "user", schema)
    assert first == second == {"grade": 2}
    assert len(calls) == 1 and calls[0]["model"] == "qwen3.5:9b-q4_K_M"
    ledger = [json.loads(line) for line in (tmp_path / "usage.jsonl").read_text().splitlines()]
    assert ledger[0]["cost_usd"] == 0.0 and ledger[0]["input_tokens"] == 100
    offline = LLM(cache_dir=tmp_path, offline=True)
    assert offline.json("judge", "ollama/qwen3.5:9b-q4_K_M", "system", "user", schema) == {"grade": 2}
    with pytest.raises(RuntimeError):
        offline.json("judge", "ollama/qwen3.5:9b-q4_K_M", "system", "another prompt", schema)
    with pytest.raises(ValueError):
        split_model("gpt-4.1")


def test_dense_encoder_prompts_queries_only_and_caches(tmp_path, monkeypatch):
    model_dir = tmp_path / "model"
    model_dir.mkdir()
    (model_dir / "prosus-model.json").write_text(json.dumps({"repo_id": "test/model", "revision": "abc"}))
    encoder = DenseEncoder(model_dir, tmp_path / "cache", "Find food")
    seen = []

    def fake_encode(texts, prompt):
        seen.append((list(texts), prompt))
        return DummyEncoder().encode([(prompt or "") + text for text in texts])

    monkeypatch.setattr(encoder, "_encode", fake_encode)
    documents = encoder.documents(["a", "b"])
    assert seen[-1] == (["a", "b"], None)
    assert np.array_equal(encoder.documents(["a", "b"]), documents) and len(seen) == 1
    encoder.queries(["pizza"])
    assert seen[-1][1] == "Instruct: Find food\nQuery: "
    encoder.queries(["pizza"])
    assert len(seen) == 2
