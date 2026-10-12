"""Offline API contracts used by the demo frontend."""

from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from retrieval.engine import RetrievalEngine


@pytest.fixture(scope="module")
def client(tmp_path_factory):
    with pytest.MonkeyPatch.context() as patch:
        # Entering the ASGI lifespan must not load or download a semantic model.
        patch.setenv("PROSUS_AUTO_BUILD", "0")
        from app import server as app_module

        patch.setattr(app_module, "engine", RetrievalEngine(
            app_module.corpus, tmp_path_factory.mktemp("api-cache"),
        ))

        def unexpected_build():
            raise AssertionError("Offline API checks must not start model inference.")

        patch.setattr(app_module, "start_semantic_build", unexpected_build)
        patch.setattr(app_module, "MODELS", tmp_path_factory.mktemp("no-models"))
        with TestClient(app_module.app) as test_client:
            yield test_client


def test_status_and_query_counts_match_assignment(client):
    status = client.get("/api/status")
    assert status.status_code == 200
    payload = status.json()
    assert payload["counts"]["items"] == 5000
    assert payload["counts"]["queries"] == 100
    assert payload["semantic"]["state"] == "not_started"
    assert {method["id"] for method in payload["methods"]} == {"bm25", "e5", "hybrid", "harrier", "full"}
    assert next(method for method in payload["methods"] if method["id"] == "bm25")["ready"] is True
    assert all(not method["ready"] for method in payload["methods"] if method["id"] != "bm25")
    queries = client.get("/api/queries")
    assert queries.status_code == 200
    assert len(queries.json()["queries"]) == 100
    assert all(query["text"].strip() for query in queries.json()["queries"])


def test_catalog_category_contract_filter_and_pagination(client):
    first_response = client.get("/api/catalog", params={"limit": 7})
    assert first_response.status_code == 200
    first = first_response.json()
    assert first["total"] == 5000
    assert (first["offset"], first["limit"]) == (0, 7)
    assert len(first["items"]) == 7
    assert first["categories"]
    assert all(isinstance(category["name"], str) and category["count"] > 0
               for category in first["categories"])
    assert sum(category["count"] for category in first["categories"]) == 5000

    second = client.get("/api/catalog", params={"offset": 7, "limit": 7}).json()
    assert (second["offset"], second["limit"]) == (7, 7)
    assert len(second["items"]) == 7
    assert not ({item["document_id"] for item in first["items"]}
                & {item["document_id"] for item in second["items"]})

    category = next(category for category in first["categories"] if category["name"])
    filtered = client.get("/api/catalog", params={"category": category["name"], "limit": 7}).json()
    assert filtered["total"] == category["count"]
    assert all(item["taxonomy_l0"] == category["name"] for item in filtered["items"])


def test_bm25_search_returns_serializable_distinct_top_five(client):
    response = client.post("/api/search", json={"query": "pizza de forno a lenha", "method": "bm25"})
    assert response.status_code == 200
    payload = response.json()
    assert payload["method"] == "bm25"
    assert len(payload["results"]) == 5
    assert [item["rank"] for item in payload["results"]] == [1, 2, 3, 4, 5]
    assert len({item["document_id"] for item in payload["results"]}) == 5
    assert len({(item["item_id"], item["merchant_id"]) for item in payload["results"]}) == 5
    assert all(isinstance(item["score"], (int, float)) for item in payload["results"])
    assert all(value >= 0 for value in payload["timings_ms"].values())
    detail = client.get("/api/items/" + payload["results"][0]["document_id"])
    assert detail.status_code == 200
    assert detail.json()["item_id"] == payload["results"][0]["item_id"]


@pytest.mark.parametrize("body", [
    {"query": "   "},
    {"query": "pizza", "method": "missing-method"},
    {"query": "pizza", "top_k": 0},
    {"query": "pizza", "top_k": 21},
])
def test_search_rejects_invalid_requests(client, body):
    response = client.post("/api/search", json=body)
    assert response.status_code == 422
    assert response.json()["detail"]


def test_unknown_document_returns_404(client):
    response = client.get("/api/items/nonexistent-document-id")
    assert response.status_code == 404


def test_compare_preserves_ready_result_and_explicit_semantic_error(client):
    response = client.post("/api/compare", json={
        "query": "pizza de forno a lenha", "methods": ["bm25", "e5"], "top_k": 5,
    })
    assert response.status_code == 200
    runs = {run["method"]: run for run in response.json()["runs"]}
    assert set(runs) == {"bm25", "e5"}
    assert len(runs["bm25"]["results"]) == 5
    assert "error" not in runs["bm25"]
    assert runs["e5"]["error"]
    assert "results" not in runs["e5"]
    direct = client.post("/api/search", json={"query": "pizza", "method": "e5"})
    assert direct.status_code == 503
    assert direct.json()["detail"]


def test_pipeline_methods_return_ranked_listings_and_interpretation(client, monkeypatch):
    from app import server as app_module
    from retrieval.pipeline import SearchPipeline
    from test_pipeline import FakeDense, FakeLLM, FakeReranker

    fake = SearchPipeline(app_module.engine, FakeLLM(), FakeDense(), FakeReranker())
    monkeypatch.setattr(app_module, "get_pipeline", lambda: fake)
    for method in ("harrier", "full"):
        response = client.post("/api/search", json={"query": "pão de queijo", "method": method, "top_k": 5})
        assert response.status_code == 200
        body = response.json()
        assert [item["rank"] for item in body["results"]] == [1, 2, 3, 4, 5]
        assert len({(item["item_id"], item["merchant_id"]) for item in body["results"]}) == 5
    assert body["understanding"]["rewrite_pt"] == "pão de queijo mineiro"
    compared = client.post("/api/compare", json={"query": "pão de queijo", "methods": ["bm25", "full"]}).json()
    assert [run["method"] for run in compared["runs"]] == ["bm25", "full"]
