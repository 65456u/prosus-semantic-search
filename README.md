# Semantic food search — Prosus AI assignment

Ranks 5,000 iFood catalog listings for 100 Portuguese food queries. 28 retrieval
systems are compared. There are no relevance labels, so the systems are scored
by an LLM judge, and the judge is then checked against evidence it did not
produce: blind human grades, human preferences on all 100 queries, and held-out
historical searches. Retrieval, judging, and the demo run locally on a 16 GB
Apple M4 laptop. The two GPU training experiments (a fine-tuned two-tower
retriever and generative retrieval) ran on the Snellius cluster. API spend:
$0.

**Submitted rankings:** [`results/qe-hybrid-vl.tsv`](results/qe-hybrid-vl.tsv)
(top 10 per query, by `itemId` and `merchantId`), plus
[`results/hybrid-vl.tsv`](results/hybrid-vl.tsv), the best system that makes no
LLM call at query time.

## Results at a glance

Each row is one component, measured against the system it modifies.

Column sources:

- **Judge:** LLM judge, nDCG@5 over 100 queries, Holm-adjusted p.
- **Grades:** blind human 0–3 grades on 19 queries.
- **Duels:** human picks between the two systems' listings at the same rank, over 100 queries.
- **ppref:** the share of all 1,150 human preferences each system orders correctly.
- **Searches:** held-out historical searches, graded nDCG@10.

| Component | Judge | Grades | Duels (wins–losses) | ppref | Searches | Verdict |
|---|---|---|---|---|---|---|
| Stronger embedder: `harrier` vs `e5` | **+0.250** (0.003) | **+0.35** (0.001) | **100–30** (<0.001) | **+0.295** (0.001) | +0.005; descriptive **+0.024** (0.03) | **Confirmed.** The largest single gain. |
| LLM query understanding: `qe-hybrid-neutral` vs `hybrid` | **+0.090** (0.003) | +0.02 (0.68) | 44–31 (0.17); rank 1: **21–9** (0.04) | **+0.062** (0.010) | random sample: **−0.018** MRR (0.02); descriptive: +0.003 (0.74) | **Real on descriptive queries.** Hurts brand searches, so route it. |
| Photo channel: `hybrid-vl` vs `hybrid` | **+0.060** (0.003) | **+0.12** (0.001) | 44–35 (0.37) | +0.045 (0.35) | +0.002 (0.37) | **Probably a small gain.** Not established on 100 queries. |
| Photos on top of query understanding: `qe-hybrid-vl` vs `qe-hybrid-neutral` | +0.027 (0.36) | +0.05 (0.16) | — | **+0.059** (0.022) | — | **Best system** under the judge and under human preferences. |
| Cross-encoder reranker: `qe-hybrid-rerank` vs `qe-hybrid` | −0.004 | +0.03 (0.54) | 57–57 | +0.009 | — | **No gain.** Adds 5 s. |
| BM25 + dense fusion: `hybrid` vs `harrier` | +0.005 | −0.05 (0.28) | 48–46 | −0.013 | **+0.009** (0.002) | **Helps brand and product searches only.** |
| Two-tower fine-tuned on search logs: `harrier-ft` vs `harrier` | +0.034 (0.13) | — | 54–42 (0.26) | +0.013 | **+0.020** (<0.001) | **Helps searches like the logs**, not these queries. |
| Past search terms written into listings: `hybrid-logs` vs `hybrid` | −0.007 | — | — | +0.040 (0.27) | **+0.095** (<0.001) | **Head queries only.** |
| Generative retrieval: `gr-tu-clean-long` vs `harrier` | **−0.180** (<0.001) | — | — | — | — | **Far behind** at this data scale. |

How to read it:

- p-values are in brackets. They are Holm-adjusted for the judge and ppref columns, raw elsewhere.
- Bold marks a difference that holds after correction.
- "Descriptive" is the 11% of held-out searches that describe a dish rather than name a brand or product; that is the kind of query the evaluation set contains.
- ppref draws on more pairs than the duels of one comparison, so it can resolve a difference the duels alone cannot.

**Recommendation.**

- **For these queries: `qe-hybrid-vl`.** It is stemmed BM25 + harrier + a Qwen3-VL text+photo channel, with local LLM query understanding.
  - It scores highest under the judge: nDCG@5 0.585 [0.54, 0.63], against 0.303 for BM25.
  - It also scores highest under human preferences: ppref 0.72, the highest of all 28 systems.
  - Each of its two additions is supported by human evidence. Query understanding is supported by the preferences and by the rank-1 duels. The photo channel on top of it is supported by the preferences.
- **Without an LLM at query time: `hybrid-vl`.** It takes 123 ms per query on the laptop. An unseen query costs 7 s with the LLM step.
- **In production: route by query kind.**
  - 89% of held-out real searches name a brand or a product (navigational 61%, generic 28%). On these, the LLM rewrite hurts, and logs-based signals help most: the hybrid with past search terms, or the fine-tuned tower.
  - Descriptive long-tail queries like the 100 here go through query understanding and the photo channel.
  - Routing is a design proposal. It was not evaluated end to end, because nearly all 100 evaluation queries are descriptive.

## Repository layout

```
data/              the two assignment CSVs go here (confidential, not in Git)
results/           submitted top-10 rankings per query (IDs only)
core/              shared code: corpus loading, cached LLM client, local models, listing photos, paths
retrieval/         every retrieval method
  engine.py          BM25 (plain and Portuguese-stemmed), e5, reciprocal-rank fusion
  pipeline.py        the ablation ladder and LLM query understanding
  multimodal.py      Qwen3-VL text+photo channel (vl-*, hybrid-vl, qe-hybrid-vl)
  two_tower.py       two-tower retriever fine-tuned on search logs (InfoNCE)
  log_expansion.py   past search terms written into the listing text
  generative/        generative retrieval: identifiers, T5Gemma 2 training, trie decoding
evaluation/        system runs, pooled LLM judge, metrics and significance tests, preference
                   and behavioural evaluation, the report, results export
tools/
  annotation/        blind labelling pages: grades, retest, listing pairs, rankings, duels
  data_explorer/     browse all listings and queries (Chinese interface)
app/               demo: FastAPI service and React UI
scripts/snellius/  Slurm scripts for the GPU training runs
tests/             offline unit tests (no model downloads)
docs/              approach, evaluation, experiments, reproduction, slides
```

## Quick start

Requires Python 3.12, `uv`, and [Ollama](https://ollama.com). Place the two assignment CSVs in
[`data/`](data).

```sh
uv venv --python 3.12 .venv
uv pip install --python .venv/bin/python -r requirements.lock.txt
ollama serve &                                  # local LLM server
.venv/bin/python core/download_model.py         # pinned HF models + Ollama models (~20 GB)
.venv/bin/python -m evaluation.runs             # ladder systems x 100 queries -> output/eval/runs.jsonl
.venv/bin/python -m evaluation.judge            # pooled blind LLM judging
.venv/bin/python -m evaluation.report           # output/eval/report.md
.venv/bin/python -m evaluation.export_results   # results/<system>.tsv
.venv/bin/python -m pytest -q
```

Every experiment, labelling tool, and GPU job is in [docs/reproduce.md](docs/reproduce.md). The
demo app: [app/README.md](app/README.md). The data explorer:
[tools/data_explorer/README.md](tools/data_explorer/README.md).

## Documentation

- [docs/approach.md](docs/approach.md): what the data showed, the systems, assumptions, and what was informed by the evaluation queries
- [docs/evaluation.md](docs/evaluation.md): methodology, full results, judge validation, human preferences, behavioural evaluation, limitations
- [docs/experiments.md](docs/experiments.md): multimodal embedder, two-tower fine-tuning, search logs, generative retrieval
- [docs/reproduce.md](docs/reproduce.md): every command
- [docs/slides.pdf](docs/slides.pdf): the five presentation slides

## Confidentiality

The dataset, the brief, and this solution are confidential to Prosus AI. Keep this repository
private. The CSVs, the brief, generated outputs, caches, and models are excluded by `.gitignore`;
`results/` holds IDs only.
