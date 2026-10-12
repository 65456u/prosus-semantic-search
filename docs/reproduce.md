# Reproduce

Every command, from setup to the GPU jobs. Run from the repository root.

Requires Python 3.12, `uv`, and [Ollama](https://ollama.com) (`brew install ollama`).
Place `5k_items_curated.csv` and `queries.csv` in `data/`.

```sh
uv venv --python 3.12 .venv
uv pip install --python .venv/bin/python -r requirements.lock.txt
ollama serve &                                  # local LLM server
.venv/bin/python core/download_model.py                     # pinned HF models + Ollama models (~20 GB)
.venv/bin/python -m evaluation.runs                         # ladder systems x 100 queries -> output/eval/runs.jsonl
.venv/bin/python -m evaluation.judge                        # pooled blind judging -> output/eval/judgments.jsonl
.venv/bin/python -m evaluation.report                       # output/eval/report.md
.venv/bin/python -m evaluation.export_results               # results/<system>.tsv

# Experiments (each adds its runs; rerun evaluation.judge for the new pool, then evaluation.report)
.venv/bin/python -m core.images                             # listing photos (~75 MB)
.venv/bin/python core/download_model.py --only qwen3-vl-embedding-2b
.venv/bin/python -m retrieval.multimodal                    # Qwen3-VL-Embedding-2B: text, photo, text+photo, hybrids
.venv/bin/python -m retrieval.log_expansion # past search terms written into listing text
.venv/bin/python -m retrieval.two_tower train --data logs --epochs 5   # GPU; then copy the model back
.venv/bin/python -m retrieval.two_tower evaluate
.venv/bin/python -m evaluation.behavioral                   # held-out search terms (recall-style)
.venv/bin/python -m evaluation.behavioral_graded # typed-only, graded, by kind of search
.venv/bin/python -m evaluation.latency                      # controlled single-request latency
.venv/bin/python -m evaluation.judge_mm                     # the judge with the listing photo

# Human evidence (each tool serves a local page and appends labels to output/eval/)
.venv/bin/python -m tools.annotation.annotate prepare && .venv/bin/python -m tools.annotation.annotate serve   # 0–3 grades
.venv/bin/python -m tools.annotation.annotate retest --pairs 40 && .venv/bin/python -m tools.annotation.annotate serve --retest --port 8767
.venv/bin/python -m tools.annotation.pairwise prepare-items && .venv/bin/python -m tools.annotation.pairwise serve --items   # listing pairs
.venv/bin/python -m tools.annotation.listwise prepare && .venv/bin/python -m tools.annotation.listwise serve   # ranking + divider
.venv/bin/python -m tools.annotation.pairwise prepare-duels && .venv/bin/python -m tools.annotation.pairwise serve --duels   # same-rank duels
.venv/bin/python -m tools.annotation.pairwise export-offline   # or label offline: a self-contained page, then import-offline "<code>"
.venv/bin/python -m tools.annotation.pairwise summary-duels
.venv/bin/python -m evaluation.preference_eval              # ppref, compatibility, power -> output/eval/preference_eval.md

# Generative retrieval (T5Gemma 2 is gated: accept its terms, provide HF_TOKEN or HF_TOKEN_PATH)
.venv/bin/python -m retrieval.generative.prepare_items      # Gemma: clean identifier phrases + training queries
.venv/bin/python -m retrieval.generative.ids                # identifier variants + diagnostics
.venv/bin/python -m retrieval.generative.ids_mm             # photo-aware semantic identifiers (after vl_baseline)
.venv/bin/python -m retrieval.generative.train --variant semantic --epochs 2 --stop-after-epoch 1
.venv/bin/python -m retrieval.generative.run --variants semantic   # decode the 100 queries -> runs.jsonl
.venv/bin/python -m pytest -q
```

GPU runs go through [`scripts/snellius/`](../scripts/snellius):

- `push.sh` copies code, data, and base models to private scratch and builds the environment.
- `submit.sh` sends a Slurm array to the A100 and H100 partitions at once; each task keeps whichever copy starts first.
- `pull.sh` fetches checkpoints for decoding on the laptop.

All LLM responses, embeddings, and reranker scores are cached by exact input,
so reruns are free and reproduce the same rankings. Model revisions are pinned
in [`core/download_model.py`](../core/download_model.py). The LLM client also
accepts OpenAI models (`openai/gpt-4.1-mini` with `OPENAI_API_KEY` in a
Git-ignored `.env`); switching providers is a configuration change.

An interactive comparison app (FastAPI + React) is described in
[`app/README.md`](../app/README.md).
