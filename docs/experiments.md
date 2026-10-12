# Experiments

Four experiments beyond the ablation ladder, each compared with the system it modifies.

## An industry-style multimodal embedder

Production food search has moved to two-tower embedders with an LLM backbone
that also read the item photo; Uber Eats, DoorDash, and Meituan describe this
direction. **Qwen3-VL-Embedding-2B** (Apache 2.0, January 2026) is the open
model of that kind that still fits on the demo laptop.

- It is used zero-shot, with the same English task instruction for queries as harrier.
- 4,795 of 5,000 listings have a photo; the others keep their text vector.
- Photos are the CDN's 300-pixel versions.

Code: [`retrieval/multimodal.py`](../retrieval/multimodal.py).

| System | nDCG@5 | Compared with | Δ | p (Holm) |
|---|---|---|---|---|
| `vl-text` | 0.473 [0.43, 0.52] | `harrier` (0.464) | +0.009 | 0.71 (1.00) |
| `vl-image` | 0.304 [0.27, 0.34] | `vl-text` | −0.169 | <0.001 (0.003) |
| `vl-fused` | 0.513 [0.47, 0.56] | `vl-text` | +0.040 | 0.01 (0.21) |
| `hybrid-vl` | 0.529 [0.48, 0.58] | `hybrid` (0.469) | +0.060 | <0.001 (0.003) |
| `qe-hybrid-vl` | 0.585 [0.54, 0.63] | `qe-hybrid-neutral` (0.559) | +0.027 | 0.03 (0.36) |

- **A bigger, LLM-backbone text embedder is not better here.** On text alone,
  the 2B model ties the 0.6B harrier.
- **The photo alone is a real but weak signal.** Retrieving by photo only
  matches stemmed BM25 (0.304 against 0.325, p = 0.48).
- **Text and photo together beat text alone,** and as a third channel in the
  hybrid they add +0.060 under the judge.
  - `hybrid-vl` is statistically indistinguishable from LLM query understanding (−0.014 against `qe-hybrid`, p = 0.38).
  - It needs 123 ms per query instead of a 7 s LLM call for an unseen query.
- **Photos and query understanding add up.** `qe-hybrid-vl` is the best system,
  both under the judge and under human preferences. It is +0.056 over
  `hybrid-vl` under the judge (p < 0.001).
- **The shared model family, checked.** The judge (Qwen3.5) could prefer its
  own family's embedder.
  - Human grades on 19 queries show a larger photo gain than the judge (+0.12 against +0.07).
  - The 100-query duels lean the same way but are not significant (44–35).
  - On held-out searches, mostly brand and product names, the photo channel changes nothing.
  - The photo gain is therefore probably real but small. It is the least certain of the recommended components.

## The industry baseline, a two-tower retriever fine-tuned on search logs

Production two-tower retrievers are trained on the platform's own engagement
logs with a contrastive loss over in-batch negatives. Here, both towers share
harrier-oss-v1-0.6b, so the result compares directly with `harrier` and still
runs on the laptop.

Training setup:

- **Training pairs:** 12,840 (search term, listing text) pairs from the 80% training split of `itemProfile.search`. The held-out 20% never enters training.
- **Batches:** no duplicate terms or listings in a batch, so in-batch negatives are true negatives.
- **Loss and optimiser:** InfoNCE with scale 20, AdamW at 2e-5, for 5 epochs on an A100.
- **Epoch selection:** by recall on 5% of the training terms, which are kept aside.
- **Second variant:** `harrier-ft-syn` adds three LLM-written customer queries per listing (doc2query-style), for 26,796 pairs.

Code: [`retrieval/two_tower.py`](../retrieval/two_tower.py).

| System | Judge nDCG@5, 100 queries | Held-out searches, nDCG@10 (1,566 terms) | Duels vs `harrier` |
|---|---|---|---|
| `harrier` | 0.464 | 0.088 | — |
| `harrier-ft` | 0.498 (+0.034, p = 0.13) | 0.109 (+0.020, p < 0.001) | 54–42 (p = 0.26) |
| `harrier-ft-syn` | 0.485 | 0.101 | — |
| `hybrid-ft` | 0.474 (vs `hybrid` +0.004) | 0.106 (vs `hybrid` +0.009, p < 0.001) | 34–42 vs `hybrid` |

- **Fine-tuning helps on the distribution it was trained on.** Held-out searches gain +0.020 nDCG@10, including descriptive ones (+0.024, p = 0.01).
- **It leaves the 100 evaluation queries statistically unchanged** under the judge, under human duels, and under ppref. The logs contain short names, and the evaluation queries are long translated descriptions.
- **Synthetic LLM queries did not help** (`harrier-ft-syn` is below `harrier-ft` on both).

## Search behaviour in the listing representation

A common production step is document expansion from logs: append the terms
users searched before reaching a listing. Here, each listing gets its 10 most
frequent past search terms ("Buscas: …"), taken only from the 80% training
split. The hashed 20% held-out split never enters a listing (unit-tested). 3,741
of 5,000 listings have training-split terms. Code:
[`retrieval/log_expansion.py`](../retrieval/log_expansion.py).

| Comparison | Judge nDCG@5, 100 evaluation queries | Held-out searches, nDCG@10 (1,566 terms) |
|---|---|---|
| `harrier-logs` vs `harrier` | 0.436 vs 0.464 (−0.028, p = 0.06) | 0.133 vs 0.088 (+0.045, p < 0.001) |
| `hybrid-logs` vs `hybrid` | 0.462 vs 0.469 (−0.007, p = 0.57) | 0.192 vs 0.097 (+0.095, p < 0.001) |

Logs make a listing look like the searches that led to it.

- **On searches that resemble past ones,** this doubles the hybrid's score. Those are short and often navigational: brands, product names.
- **On the 100 evaluation queries it gives nothing.** None of them occurs among the training terms.
- **The held-out gain is an upper bound for genuinely new searches.** Held-out terms are often near-duplicates of training terms (*acai* held out, *acai da cidade* in training).

## Generative retrieval

A seq2seq model, **T5Gemma 2 (270M-270M)**, is fine-tuned to read a query and
spell the identifier of the matching item directly. It decodes with a prefix
trie, so only real identifiers can be produced (beam 20). This is the retrieval
half of the "one generative model for search and recommendation" direction
(semantic IDs, as in TIGER, OneRec, and Spotify's joint search and
recommendation work), scaled down.

**Identifier design was decided with data before training.** These offline
diagnostics are on the 4,672 distinct product contents (identical listings share
one identifier):

| Identifier | Collisions | Length | Note |
|---|---|---|---|
| Global RQ-KMeans, 3 x 64 codes (industry default) | 50% | 3 codes | residual codebooks cannot separate a 5k catalog |
| Hierarchical k-means 64 → 16 → 16 | 4.7% | 3 codes | balanced and semantic, but opaque tokens |
| PAMT-style TU: name, menu section, taxonomy | 1.2% | 28 tokens | first word is near-random (1,378 values); names are messy |
| Keyword-labelled cluster tree | 4.7% | 6 tokens | readable, but cluster words mislabel items |

Listing names are poor identifiers ("2115 - salmão…", "PEQUENA (4 PEDAÇOS)",
"Ninho"). A local LLM (Gemma 4 E4B) therefore first rewrites every content as
*tipo · variante · detalhe*. Any word it adds must occur in the listing text;
otherwise the item is retried, and finally falls back to its cleaned name.

Four identifier variants are compared:

- `gr-tu-raw`: `name | menu section | taxonomy`.
- `gr-tu-clean`: the same, with the LLM-normalised name.
- `gr-semantic`: `LLM-named k-means cluster | canonical kind | variant detail`.
- `gr-semantic-mm`: the same, with clusters built from harrier text vectors concatenated with Qwen3-VL text+photo vectors.

**Training data.** No evaluation query is used:

- product text and name → identifier;
- three LLM-written customer queries per item;
- historical search terms (38,497 training pairs in all).

A hashed 5% of the query-style pairs is the development set. Code:
[`retrieval/generative/`](../retrieval/generative); Slurm scripts in
[`scripts/snellius/`](../scripts/snellius).

Judge results, 100 queries:

| System | Epochs (best) | nDCG@5 | Hit@5 | Dev Recall@10 | Most repeated listing |
|---|---|---|---|---|---|
| `gr-tu-raw` | 1 | 0.148 [0.11, 0.19] | 0.20 | 0.07 | 12 queries |
| `gr-tu-clean` | 1 | 0.237 [0.19, 0.28] | 0.37 | 0.19 | 13 queries |
| `gr-semantic` | 1 | 0.189 [0.15, 0.23] | 0.37 | 0.13 | 35 queries |
| `gr-semantic-mm` | 1 | 0.227 [0.19, 0.27] | 0.40 | 0.14 | 38 queries |
| `gr-tu-clean-long` | 8 (4) | **0.284** [0.24, 0.33] | 0.47 | 0.21 | 8 queries |
| `gr-semantic-long` | 8 (6) | 0.271 [0.23, 0.32] | **0.51** | 0.23 | 5 queries |
| `gr-semantic-mm-long` | 8 (7) | 0.265 [0.22, 0.32] | 0.45 | 0.23 | 7 queries |
| `gr-tu-clean-50ep` | 50, stopped at 19 (9) | 0.238 [0.20, 0.28] | 0.37 | 0.20 | 9 queries |
| `gr-semantic-50ep` | 50, stopped at 22 (12) | 0.248 [0.20, 0.30] | 0.37 | 0.18 | 5 queries |
| `gr-semantic-mm-50ep` | 50, stopped at 19 (9) | 0.245 [0.20, 0.29] | 0.39 | 0.18 | 9 queries |
| `harrier` (reference) | — | 0.464 [0.41, 0.51] | 0.67 | — | 7 queries |

- **Cleaning the names is the biggest identifier effect.** LLM-normalised titles
  beat raw listing names by +0.090 (Holm p = 0.004).
- **Training longer does not close the gap.** Eight epochs add +0.04 to +0.08
  and cure the hubness of semantic identifiers.
  - A 50-epoch schedule with early stopping (patience 10) stopped after 19–22 epochs.
  - Its best development recall (0.18–0.20) was no better than the eight-epoch runs, and it scored lower under the judge (−0.046 for `tu-clean`, p = 0.05).
  - Checkpoints saved every five epochs were judged too. In all three variants the judge's nDCG@5 is highest at epoch 5 (0.24–0.29) and lower afterwards (by up to 0.06), while development recall stays flat or rises slightly (0.16–0.18). Longer training fits the short, log-like training queries at the expense of these descriptive queries. The table keeps the checkpoints chosen by development recall, because choosing by this curve would use the evaluation queries.
  - With about eight training examples per item, the model saturates early. The best one still trails harrier by 0.18 nDCG@5 (p < 0.001).
- **Identifier design matters early, not late.** After one epoch, photo-aware
  semantic identifiers beat text-only ones (+0.038, p = 0.02) and cut non-food
  results from 19% to 5%. After eight epochs the designs are within 0.02 of each
  other.
