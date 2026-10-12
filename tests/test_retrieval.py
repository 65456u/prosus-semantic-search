"""Meaningful offline regression checks for baseline and embedding-cache safety."""

from __future__ import annotations

import pytest

from core.corpus import Corpus
from retrieval.engine import RetrievalEngine, SemanticNotReady, reciprocal_rank_fusion

from conftest import ROOT, DummyEncoder, write_corpus


@pytest.fixture(scope="module")
def assignment_corpus():
    return Corpus(ROOT / "data" / "5k_items_curated.csv", ROOT / "data" / "queries.csv")


def result_ids(response):
    return [item["document_id"] for item in response["results"]]


def dummy_engine(corpus, cache_dir, encoder=None):
    encoder = encoder or DummyEncoder()
    engine = RetrievalEngine(corpus, cache_dir)
    engine._load_model = lambda: (encoder, encoder.variant, "cpu")
    engine.build_semantic()
    return engine, encoder


def test_source_rows_and_document_ids_are_preserved(assignment_corpus):
    assert len(assignment_corpus.items) == 5000
    assert len(assignment_corpus.queries) == 100
    assert len(assignment_corpus.document_ids) == 5000
    assert len(set(assignment_corpus.document_ids)) == 5000
    reloaded = Corpus(ROOT / "data" / "5k_items_curated.csv", ROOT / "data" / "queries.csv")
    assert reloaded.document_ids == assignment_corpus.document_ids


@pytest.mark.parametrize("query", [
    "Pizza de massa fina assada em forno a lenha",
    "Sopa de macarrão feita à mão",
    "Sorvete de morango",
    "Arroz integral",
    "Hambúrguer vegetariano",
])
def test_bm25_returns_valid_distinct_stable_top_five(assignment_corpus, tmp_path, query):
    engine = RetrievalEngine(assignment_corpus, tmp_path)
    response = engine.search(query, "bm25", top_k=5)
    assert len(response["results"]) == 5
    assert len(set(result_ids(response))) == 5
    assert set(result_ids(response)).issubset(assignment_corpus.document_ids)
    listings = [(item["item_id"], item["merchant_id"]) for item in response["results"]]
    assert len(set(listings)) == 5
    assert result_ids(engine.search(query, "bm25", top_k=5)) == result_ids(response)


def test_bm25_handles_portuguese_accents_case_and_punctuation(tiny_csvs, tmp_path):
    engine = RetrievalEngine(Corpus(*tiny_csvs), tmp_path / "cache")
    accented = engine.search("PÃO, de QUEIJO!", "bm25", top_k=3)
    plain = engine.search("pao de queijo", "bm25", top_k=3)
    assert result_ids(accented) == result_ids(plain)
    assert accented["results"][0]["name"] == "Pão de queijo"


@pytest.mark.parametrize("method", ["e5", "hybrid"])
def test_unbuilt_semantic_search_fails_explicitly(tiny_csvs, tmp_path, method):
    engine = RetrievalEngine(Corpus(*tiny_csvs), tmp_path / "cache")
    with pytest.raises(SemanticNotReady):
        engine.search("pizza", method, top_k=3)


def test_rrf_rewards_agreement_and_uses_one_based_ranks():
    fused = reciprocal_rank_fusion([[0, 1, 2], [1, 2, 3]], constant=60)
    assert sorted(fused, key=lambda document: (-fused[document], document)) == [1, 2, 0, 3]
    assert fused[0] == pytest.approx(1 / 61)
    assert fused[1] == pytest.approx(1 / 62 + 1 / 61)


def test_e5_prefixes_and_persistent_query_cache(tiny_csvs, tmp_path):
    corpus = Corpus(*tiny_csvs)
    cache_dir = tmp_path / "cache"
    engine, encoder = dummy_engine(corpus, cache_dir)
    assert len(encoder.inputs) == len(corpus.items)
    assert all(text.startswith("passage: ") for text in encoder.inputs)

    first = engine.search("pão de queijo", "e5", top_k=3)
    assert first["cache_hit"] is False
    assert encoder.inputs[-1] == "query: pão de queijo"
    inference_count = len(encoder.inputs)
    second = engine.search("pão de queijo", "e5", top_k=2)
    assert second["cache_hit"] is True
    assert len(encoder.inputs) == inference_count
    assert result_ids(second) == result_ids(first)[:2]

    reloaded, fresh_encoder = dummy_engine(corpus, cache_dir)
    assert fresh_encoder.inputs == []
    persisted = reloaded.search("pão de queijo", "e5", top_k=3)
    assert persisted["cache_hit"] is True
    assert fresh_encoder.inputs == []
    assert result_ids(persisted) == result_ids(first)


def test_bypass_query_cache_measures_real_inference(tiny_csvs, tmp_path):
    engine, encoder = dummy_engine(Corpus(*tiny_csvs), tmp_path / "cache")
    engine.search("pizza", "e5", top_k=3)
    count = len(encoder.inputs)
    response = engine.search("pizza", "e5", top_k=3, use_query_cache=False)
    assert response["cache_hit"] is False
    assert len(encoder.inputs) == count + 1
    assert encoder.inputs[-1] == "query: pizza"


def test_changed_document_content_is_reencoded(tiny_csvs, tmp_path):
    cache_dir = tmp_path / "cache"
    dummy_engine(Corpus(*tiny_csvs), cache_dir)
    changed_csvs = write_corpus(tmp_path / "changed", [
        "Queijo e massa assada com ervas novas", "Forno a lenha", "Gelado", "Pacote 500g",
    ])
    changed, encoder = dummy_engine(Corpus(*changed_csvs), cache_dir)
    assert any("ervas novas" in text for text in encoder.inputs)
    assert all(text.startswith("passage: ") for text in encoder.inputs)
    assert len(changed.search("queijo", "e5", top_k=3)["results"]) == 3


def test_changed_model_identity_invalidates_document_and_query_vectors(tiny_csvs, tmp_path):
    corpus = Corpus(*tiny_csvs)
    cache_dir = tmp_path / "cache"
    original, _ = dummy_engine(corpus, cache_dir)
    original.search("pizza", "e5", top_k=3)
    changed, encoder = dummy_engine(corpus, cache_dir, DummyEncoder("model-b"))
    assert len(encoder.inputs) == len(corpus.items)
    assert all(text.startswith("passage: ") for text in encoder.inputs)
    response = changed.search("pizza", "e5", top_k=3)
    assert response["cache_hit"] is False
    assert encoder.inputs[-1] == "query: pizza"
