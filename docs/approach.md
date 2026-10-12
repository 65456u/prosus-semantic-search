# Approach

What the data showed, the systems compared, and the assumptions behind them. Results are in [evaluation.md](evaluation.md) and [experiments.md](experiments.md).

## What the data told us first

These observations drove most design decisions. All numbers are computed from
the supplied files.

| Observation | Evidence | Consequence |
|---|---|---|
| The queries read like **literal translations of English food phrases**. | *Envoltório de giroscópio* ("gyroscope wrap" = gyro wrap), *pão de bolso* (pocket bread = pita), *frango frito ao estilo sulista* (Southern fried chicken). 51 query words longer than three letters never occur anywhere in the catalog. | Keyword and embedding matching on the raw query cannot work for these. An LLM recovers the intended dish before retrieval. Relevance is judged against that intent (see Assumptions). |
| The catalog speaks **real Brazilian menu Portuguese**. | *combo* 143 listings vs *combinado* 16; *wrap* 9 vs *envoltório* 1; *bowl* 4 vs *tigela* 1; "c/" for *com* in 30 listings. | Query rewriting asks for the words menus actually use, loanwords included. |
| **Inflection** breaks exact keyword matching. | *batata frita* (singular) in 82 listings, *batatas fritas* in 21. | A Portuguese Snowball stemmer for BM25 (`bm25-pt`). |
| Only a third of the catalog is prepared food. | 1,707 *ALIMENTOS_PREPARADOS*; 1,408 rows are health, beauty, home, pet, or electronics. One query is itself a household product (a Lysoform disinfectant). | Non-food results are reported as an error signal. The judging rubric scores "a different product than requested", not "not food". |
| Listings repeat. | 5,000 rows, 4,997 distinct (itemId, merchantId) listings; 376 product names appear more than once. | Results are distinct listings; the same product at two merchants counts as two listings. |
| Real searches are mostly names. | Of 1,566 held-out typed search terms, a local LLM classifies 61% as navigational (brand, store, medicine), 28% as generic (a plain dish or product), 11% as descriptive. | The 100 evaluation queries are the descriptive long tail. Production traffic needs a different path for names (see the routing recommendation). |

## Systems

Each ladder system adds exactly one component to the previous one, so the
evaluation measures what every component contributes.

| System | Adds | Model |
|---|---|---|
| `bm25` | accent-insensitive keyword baseline | — |
| `bm25-pt` | Portuguese stemming | Snowball (Portuguese) |
| `e5` | small dense retriever | `intfloat/multilingual-e5-small` |
| `harrier` | stronger dense retriever, instruction-prompted queries | `microsoft/harrier-oss-v1-0.6b` (MIT, 2026) |
| `hybrid` | reciprocal-rank fusion of `bm25-pt` and `harrier` | — |
| `qe-hybrid` | LLM query understanding: intent, a natural Brazilian rewrite, dish names; RRF over raw and rewritten queries | Gemma 4 E4B (local, Q4_K_M) |
| `qe-hybrid-rerank` | cross-encoder reranking of the top 30 against the query plus its interpreted intent | `nvidia/llama-nemotron-rerank-1b-v2` |

All systems use one product text: name, description, category, taxonomy, and
explicitly positive dietary attributes. Prices are not used. Code:
[`retrieval/pipeline.py`](../retrieval/pipeline.py).

Further systems test one question each against the system they modify:

| System | Question | Compared with |
|---|---|---|
| `qe-hybrid-neutral` | Does query understanding still help without the prompt hints taken from the evaluation queries? | `hybrid` |
| `vl-text`, `vl-image`, `vl-fused` | How does an LLM-backbone multimodal embedder (Qwen3-VL-Embedding-2B, 2026) do on text, on the photo alone, and on both? | `harrier`, `vl-text` |
| `hybrid-vl` | Does a text+photo channel help a production-style hybrid? | `hybrid` |
| `qe-hybrid-vl` | Are query understanding and photos complementary? | `qe-hybrid-neutral`, `hybrid-vl` |
| `harrier-ft`, `harrier-ft-syn`, `hybrid-ft` | Industry baseline: does a two-tower retriever fine-tuned on search logs (InfoNCE, in-batch negatives) beat the zero-shot one? | `harrier`, `hybrid` |
| `harrier-logs`, `hybrid-logs` | Does past search behaviour help when written into the listing text? | `harrier`, `hybrid` |
| `gr-tu-raw`, `gr-tu-clean`, `gr-semantic`, `gr-semantic-mm` | Can a small generative model retrieve by spelling item identifiers, and which identifiers work? | `harrier`, each other |
| `gr-*-long`, `gr-*-50ep` | Does more training (8 epochs; up to 50 with early stopping) close the gap? | the one-epoch run |

## Assumptions

- **Relevance follows intent, not wording.** When a query is a literal
  translation, the judge and the rewriter aim at the food the query most
  plausibly means. Ambiguous cases stay ambiguous: *sulista* can mean the US
  South (the likely English source) or southern Brazil.
- **A listing is (itemId, merchantId).** The same product from two merchants is
  two results.
- **No personalisation or popularity.** The ladder systems use no behavioural
  fields. The `-logs`, `-ft`, and generative systems use historical search
  terms from the 80% training split only. Evaluation queries are never used for
  training.
- **Top 10 per query is returned; the evaluation scores the top 5.**

## What was informed by the evaluation queries

The brief supplies 100 queries and no separate development set, and I read
them while designing. These choices were shaped by that reading:

- **The query-understanding prompt.** It names loanword examples (wrap, bowl,
  burger, combo) and says queries may be literal translations. Those examples
  match words in 15 of the 100 queries.
  - On those 15 queries, the `qe-hybrid` gain over `hybrid` is +0.117 nDCG@5 (p = 0.006). On the other 85 it is +0.067 (p < 0.001).
  - `qe-hybrid-neutral` repeats the system with both hints removed.
  - Its gain shrinks on the 15 hint-word queries (+0.084, p = 0.04) and grows on the other 85 (+0.090, p < 0.001).
  - Overall it scores slightly *higher* than `qe-hybrid` (0.559 against 0.543, p = 0.20).
  - The hints helped only the queries they were written from. The query-understanding gain does not depend on them, and the neutral prompt is the one used in the demo and in `qe-hybrid-vl`.
- **The reranker input.** Feeding the reranker the interpreted intent was decided
  after looking at four evaluation queries.
- **The judge prompt.** It shares the "judge the intended dish" instruction with
  the query-understanding prompt, which may favour `qe-hybrid`. That is one
  reason query understanding was also checked with human preferences.

No system was trained or tuned on the evaluation queries, their judgments, or
the human labels.
