# Local food search demo

React + Vite + TypeScript frontend, FastAPI service, and a shared Python retrieval core. The original Portuguese assignment data stays local. The interface is in English for presentation use.

## Open the prepared demo

Run:

```sh
python3 app/launch.py start --open
```

The demo is served at http://127.0.0.1:8000/. Closing the terminal does not stop the service. Use `python3 app/launch.py stop` to stop it. After a restart, BM25 is available while E5 loads its model and saved document vectors. The interface shows that progress.

## First-time setup on another machine

Requires Python 3.12, Node.js 22.12 or newer, `uv`, and Ollama (`brew install ollama`, then `ollama serve`).

```sh
uv venv --python 3.12 .venv
uv pip install --python .venv/bin/python -r requirements.lock.txt
.venv/bin/python core/download_model.py   # pinned HF models and the two Ollama models (~20 GB)
npm --prefix app/frontend ci
npm --prefix app/frontend run build
.venv/bin/python -m app.build_index
python3 app/launch.py start --open
```

The public E5 assets are downloaded through the Hugging Face CLI at revision `614241f622f53c4eeff9890bdc4f31cfecc418b3`. Dataset records are not sent to Hugging Face. Inference uses local model files; no OpenAI API key or API credit is used. Images are disabled by default. Enabling images requests the original assignment image URLs from iFood.

## Development

Run the Python API and Vite in separate terminals:

```sh
.venv/bin/python -m uvicorn app.server:app --host 127.0.0.1 --port 8000
npm --prefix app/frontend run dev
```

Vite proxies `/api` to the local Python service. The built frontend is served directly by FastAPI, so the prepared demo needs one running service.

## Implemented methods

- **BM25:** accent-insensitive Portuguese tokenization, a small documented stopword list, `k1=1.5`, `b=0.75`. Negation is retained. If fewer than five listings have positive scores, zero-score ties fill the remaining ranks and the response includes a warning.
- **Multilingual E5:** `intfloat/multilingual-e5-small`, `query: ` and `passage: ` prefixes, normalized 384-dimensional vectors, exact cosine retrieval. Encoding runs locally. Device selection prefers CUDA, then MPS, then CPU; actual device is reported by the service.
- **Hybrid (demo):** reciprocal-rank fusion (`k=60`) of the top 100 distinct positive-score BM25 listings and top 100 distinct E5 listings. It reuses the same E5 vectors. This is the demo's quick hybrid; the evaluated `hybrid` fuses stemmed BM25 with harrier.
- **Harrier:** `microsoft/harrier-oss-v1-0.6b` with an instruction-prompted query, served by the evaluated pipeline (`retrieval/pipeline.py`, system `harrier`).
- **LLM + hybrid:** the evaluated `qe-hybrid-neutral`: Gemma 4 E4B (local, via Ollama) interprets the query with the neutral prompt, then stemmed BM25 and harrier retrieve for the raw and rewritten query, fused with RRF. The interpreted intent is shown above the results. The 100 assignment queries are cached; a new query needs `ollama serve` running and takes about 7 s on an M4 for the LLM step. The photo channel of the recommended `qe-hybrid-vl` is not in the demo; its rankings are in `results/`.

The Compare view shows BM25, Harrier, and LLM + hybrid side by side. See the repository README for the evaluation.

All methods use one common product text: supplied name, description, category, taxonomy, and positive product attributes. Prices and behavioral histories are not encoded. False dietary flags are not converted into ingredient claims. Retrieval returns distinct `(itemId, merchantId)` listings; catalog browsing retains all 5,000 source rows. Row-based document IDs allow every result to link back to its exact source record.

The dataset has no provided query relevance labels. Scores are ranking scores, not probabilities or validated quality measurements. Different methods' raw scores are not directly comparable.

## Reuse and cache behavior

`core/corpus.py` and `retrieval/engine.py` are independent of React and FastAPI. The API, batch export, and future evaluation scripts share this core.

- Model assets: `.demo-cache/models/`.
- Per-input document and query vectors: `.demo-cache/retrieval/docs.sqlite3` and `queries.sqlite3`.
- Document matrix snapshots: `.demo-cache/retrieval/doc-embeddings/`.

Cache keys include the model revision, local model file digest, exact prefixed input text, and encoding configuration. Changed product text reuses unaffected vectors. Different models have distinct vector spaces and caches. Search results are scored live; `cache_hit` means query-vector reuse, not a saved ranking. Query-cache bypass disables both reads and writes.

## Validation and export

```sh
.venv/bin/python -m pytest tests -q
.venv/bin/python -m app.run_queries
```

The second command calls the running service for all 100 queries and all three methods, saves Top 5 results, and checks valid distinct listings. Its output is functional validation, not a relevance evaluation. Add `--no-query-cache` to exercise fresh query encoding.

Displayed server timings separate query encoding and retrieval. They exclude HTTP transport, browser rendering, image loading, and initial model/index preparation. Comparison requests run the methods serially; later methods may reuse a query vector, which is explicitly marked. A controlled latency/throughput benchmark should measure the API separately with fixed concurrency and report cache conditions.

## Local files and confidentiality

Original CSV and PDF files are unchanged. Data, vectors, logs, results, dependencies, and model assets are excluded from Git by the project `.gitignore`. Follow the assignment's confidentiality terms when sharing the code or any derived artifacts. This demo binds to `127.0.0.1` and is not publicly deployed.

The service log is `.demo-cache/runtime/server.log`. `python3 app/launch.py status` reports current readiness. Set `PROSUS_DEVICE=cpu` to force CPU, or `PROSUS_AUTO_BUILD=0` to start only BM25 until the UI requests model preparation.
