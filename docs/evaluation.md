# Evaluation

How the 28 systems are scored without relevance labels, how the LLM judge is checked, and the full results.

## Evaluation methodology

There are no relevance labels. The evaluation creates them with an LLM judge,
then tests the judge against evidence that does not come from it.

**The judge.** For every query, the top 5 of every system are merged
(TREC-style pooling: 6,761 (query, listing) pairs). All systems are scored on
the same judged items, and every system's top 5 is fully judged. A local
**Qwen3.5-9B** grades each pair on its own on a 0–3 scale: exact, good
substitute, weak, or irrelevant. The judge never sees the producing system, the
rank, or the other candidates.

The judge is from a different model family than the main systems' components:
the Gemma rewriter, the Llama-based reranker, and the e5 and harrier encoders.
This limits self-preference. The exception is the Qwen3-VL embedder in the
photo experiments: it shares the judge's family, so those gains are also
checked against human evidence.

**Evidence the judge is checked against:**

| Evidence | Size | What it adds |
|---|---|---|
| Human grades, same 0–3 scale | 381 pairs, 20 queries stratified by query type | agreement with the judge; system ordering under human grades |
| Human retest, blind second pass | 40 pairs, balanced across grades | the annotator's self-consistency: the ceiling any judge can reach |
| Human duels: the two systems' listings at the same rank, side by side | 862 duels over all 100 queries, 8 system comparisons | human evidence on every query, spent only where two systems differ |
| Human preferences between two listings | 45 pairs: graded equal, graded in conflict with the judge, clearly different | where the judge is biased rather than noisy |
| Human ranking of 8 listings plus an "acceptable" divider | 20 queries | order and acceptability in context |
| Historical search behaviour | 1,566 held-out typed search terms | LLM-free relevance: the listings users reached, by how often |
| Judge, side by side | 195 list pairs, both orders | list-level preference, and how often the verdict survives swapping the order |
| Judge with the listing photo | 357 human-labelled pairs | whether seeing the item changes the judgment |

The human labelling pages show the Portuguese text with Chinese and English
translations made by Gemma (not by the judge). They never show the system or
the rank, and left and right are randomised.

**Metrics.**

- **Judge grades:** nDCG@5 is primary. Its ideal comes from all judged listings for the query, so every system is divided by the same ideal. Precision@5 and Hit@5 at grade ≥ 2, and MRR@5, are secondary.
- **Human preferences:** ppref (Carterette and Bennett, 2008) is the share of human preferences a system's top 10 orders correctly.
- **Human rankings:** compatibility (Clarke et al., 2021) is the rank-biased overlap with the annotator's ideal ranking.

**Statistics.**

- Every number has a 95% bootstrap confidence interval over queries.
- Each comparison is a paired permutation test. Duels use a sign test.
- p-values are Holm-adjusted across each family of comparisons.
- Each judge comparison reports the smallest difference it could detect at 80% power.

**Diagnostics:**

- the catalog ceiling: queries with any acceptable listing in the pool;
- results by query type and by vocabulary overlap with the catalog;
- the non-food rate in the top 5;
- the most repeated listing per system (hubness);
- whether a low reranker score flags queries with no good match.

Code: [`evaluation/`](../evaluation). Metric and agreement functions are unit-tested
against hand-computed values and SciPy
([`tests/test_metrics.py`](../tests/test_metrics.py)).

## Results

LLM-judged, 100 queries, mean with 95% bootstrap CI. Δ is the paired
difference in nDCG@5 against the row above. p is from a permutation test; the
Holm-adjusted p is in brackets, adjusted over every comparison in the full
report.

| System | nDCG@5 | P@5 | MRR@5 | Hit@5 | Δ nDCG@5 | p (Holm) |
|---|---|---|---|---|---|---|
| `bm25` | 0.303 [0.25, 0.35] | 0.208 | 0.439 | 0.55 | — | — |
| `bm25-pt` | 0.325 [0.28, 0.37] | 0.228 | 0.411 | 0.58 | +0.022 | 0.17 (1.00) |
| `e5` | 0.214 [0.17, 0.26] | 0.142 | 0.296 | 0.44 | −0.111 | <0.001 (0.003) |
| `harrier` | 0.464 [0.41, 0.51] | 0.322 | 0.520 | 0.67 | +0.250 | <0.001 (0.003) |
| `hybrid` | 0.469 [0.42, 0.52] | 0.318 | 0.538 | 0.73 | +0.005 | 0.75 (1.00) |
| `qe-hybrid` | 0.543 [0.50, 0.59] | 0.384 | 0.571 | 0.78 | +0.074 | <0.001 (0.003) |
| `qe-hybrid-rerank` | 0.539 [0.49, 0.59] | 0.378 | 0.595 | 0.76 | −0.004 | 0.82 (1.00) |

The other systems, against the same pool:

| System | nDCG@5 | P@5 | MRR@5 | Hit@5 | ppref (human) |
|---|---|---|---|---|---|
| `qe-hybrid-neutral` | 0.559 [0.51, 0.60] | 0.406 | 0.606 | 0.78 | 0.659 |
| `hybrid-vl` | 0.529 [0.48, 0.58] | 0.368 | 0.586 | 0.74 | 0.626 |
| **`qe-hybrid-vl`** | **0.585 [0.54, 0.63]** | **0.428** | **0.606** | 0.76 | **0.721** |
| `harrier-ft` | 0.498 [0.45, 0.55] | 0.354 | 0.609 | 0.72 | 0.613 |
| `hybrid-logs` | 0.462 [0.41, 0.51] | 0.328 | 0.534 | 0.72 | 0.633 |
| `gr-tu-clean-long` (best generative) | 0.284 [0.24, 0.33] | 0.196 | 0.356 | 0.47 | 0.446 |

All 28 systems are in `output/eval/report.md` (judge) and
`output/eval/preference_eval.md` (human preferences).

What the numbers say:

- **The catalog limits every system.** 92 of 100 queries have an acceptable
  listing somewhere among the 6,761 judged candidates, and 63 have an exact
  match. A gyro wrap or a pita simply is not on sale.
- **The small dense model is worse than keywords.** e5-small collapses onto hub
  items: one chocolate bar appears in the top 5 of 25 different queries.
  harrier-0.6b fixes this and is the largest single gain.
- **Query understanding helps most where wording is weakest.** Its nDCG gain by
  query type: +0.12 on meal or vague queries, +0.10 on cuisine or style, +0.06
  on dish-with-modifier, and +0.04 on specific dishes.
- **Reranking is a wash.** The cross-encoder improves MRR and dish-with-modifier
  queries (+0.02) but hurts cuisine and style queries (−0.07). It was trained for
  question–passage retrieval, and menu entries are short. A low reranker score
  only weakly flags queries without a good match (Hit@5 0.71 vs 0.89).
- **Latency on an M4 laptop** (one request at a time, fresh query vectors):
  - `hybrid-vl`: 123 ms.
  - `qe-hybrid`: 96 ms of retrieval. A query the LLM has not seen adds 7.0 s (median, Gemma 4 E4B, Q4).
  - Reranking 30 candidates adds another 5.1 s.
- **Cost:** $0 in API spend. All LLM work ran on the laptop: about 10,000 Qwen judge calls and 4,500 Gemma calls.

Known rewriter errors, from reading all 100 interpretations:

- *frango frito ao estilo sulista* and *comida caseira no estilo do Centro-Oeste* were read as Brazilian regions. Their likely English sources are Southern (US) fried chicken and Midwestern home cooking.
- The same model read *tigela estilo sudoeste* correctly as a Southwestern (US) bowl.

### Judge validation against human labels

One annotator labelled 356 of the 357 pooled listings for 20 stratified queries,
blind to system and rank, with Gemma translations next to the Portuguese text.
A second round added the 25 listings that three later systems
(`qe-hybrid-neutral`, `hybrid-vl`, `qe-hybrid-vl`) brought into the same queries,
again blind to system; the annotator did know these were new listings. 16 pairs
were marked unsure and excluded.

| Agreement (365 pairs) | Value |
|---|---|
| Exact grade | 50.4% |
| Within one grade | 84.7% |
| Quadratic-weighted κ (0–3) | 0.55 |
| κ on acceptable (≥ 2) vs not | 0.36 |
| Kendall τ of the ten-system ordering (19 fully labelled queries) | 0.64 |

**How good can a judge get?** The same annotator relabelled 40 of these pairs
blind, a few hours after the first pass, sampled evenly across first-pass
grades. All rows below use the same 40 pairs, because κ depends on the grade
mix. Brackets are 95% bootstrap CIs.

| Same 40 pairs | Exact | Within one | Weighted κ | κ acceptable |
|---|---|---|---|---|
| Annotator, pass 1 vs pass 2 (ceiling) | 52% | 92% | 0.71 [0.53, 0.83] | 0.60 [0.35, 0.80] |
| Annotator pass 1 vs judge | 50% | 85% | 0.62 [0.42, 0.78] | 0.40 [0.14, 0.65] |

On graded relevance, the judge agrees with the annotator about as well as the
annotator agrees with themselves. The weighted-κ gap is +0.09, with a 95% CI of
[−0.13, +0.31].

The weak spot is the acceptable-or-not boundary: κ 0.40 against 0.60. Grade 2
is unstable even for the annotator: 6 of 10 first-pass 2s changed on retest.
Metrics that threshold at grade 2 (P@5, Hit@5) are therefore noisier than
graded nDCG@5, which is why nDCG@5 is the primary metric.

Paired nDCG@5 differences on the same 19 queries, mean [95% bootstrap CI]:

| Comparison | Human grades | Judge grades |
|---|---|---|
| `harrier` − `e5` | +0.35 [+0.18, +0.50], p = 0.001 | +0.30 [+0.16, +0.42], p = 0.001 |
| `hybrid` − `harrier` | −0.05 [−0.13, +0.03], p = 0.28 | +0.04 [−0.01, +0.09], p = 0.16 |
| `qe-hybrid` − `hybrid` | +0.04 [−0.04, +0.12], p = 0.42 | +0.07 [+0.00, +0.15], p = 0.08 |
| `qe-hybrid-rerank` − `qe-hybrid` | +0.03 [−0.05, +0.13], p = 0.54 | −0.00 [−0.06, +0.05], p = 0.91 |
| `qe-hybrid-neutral` − `hybrid` | +0.02 [−0.07, +0.11], p = 0.68 | +0.06 [−0.01, +0.14], p = 0.15 |
| `hybrid-vl` − `hybrid` | +0.12 [+0.05, +0.19], p = 0.001 | +0.07 [+0.01, +0.13], p = 0.04 |
| `qe-hybrid-vl` − `qe-hybrid-neutral` | +0.05 [−0.01, +0.10], p = 0.16 | +0.02 [−0.01, +0.06], p = 0.20 |

How the judge disagrees:

- **It is about one grade stricter on relevant listings** (human 3 → judge 1 in 26 pairs). It penalises a missing requested style even when the dish matches: smoked turkey or BBQ pork for *prato de churrasco defumado*.
- **It makes knowledge and reading errors.** It called *cupim* (beef hump) "oxtail" and read *saboroso* (savoury) as "sweet".

**Statistical power.** The per-query spread of paired differences between the
strong systems is SD 0.15–0.21 under the judge. With 100 queries, the judge can
therefore resolve differences of about 0.04–0.06 nDCG@5 at 80% power. The 19
human-graded queries can resolve only 0.12–0.17. This gap is why the human
evidence was extended to all 100 queries with duels.

### Human preferences on all 100 queries

A 0–3 grade is coarse. On the 45 listing pairs below, the annotator still
preferred one listing in 13 of 15 pairs they had graded equally. Preferences
are finer and quick to give (about 6 s per pair), so the human evidence was
extended to all 100 queries with preferences.

**Duels.** For each of 8 system comparisons and each of the 100 queries, the two
systems' listings at the same rank (ranks 1 and 2) are shown side by side. They
are shown blind, in random order, and identical listings are skipped. The
annotator picks the better listing, "about the same", or "both bad". 862 duels
were labelled, all by the same annotator
([`tools/annotation/pairwise.py`](../tools/annotation/pairwise.py) `prepare-duels`).

| Comparison | Wins | Losses | Ties (both bad) | Sign test p |
|---|---|---|---|---|
| `harrier` vs `e5` | 100 | 30 | 61 (47) | <0.001 |
| `qe-hybrid-neutral` vs `hybrid` | 44 | 31 | 37 (14) | 0.17 (rank 1 alone: 21–9, p = 0.04) |
| `qe-hybrid-vl` vs `hybrid-vl` | 46 | 36 | 39 (20) | 0.32 |
| `hybrid-vl` vs `hybrid` | 44 | 35 | 42 (25) | 0.37 |
| `harrier-ft` vs `harrier` | 54 | 42 | 70 (38) | 0.26 |
| `hybrid` vs `harrier` | 48 | 46 | 62 (34) | 0.92 |
| `qe-hybrid-rerank` vs `qe-hybrid` | 57 | 57 | 60 (24) | 1.00 |
| `hybrid-ft` vs `hybrid` | 34 | 42 | 60 (36) | 0.42 |

**Preference metrics over all human evidence.** The listwise orders, listing
pairs, and duels together give 1,150 human preferences. ppref scores each
system by the share of those preferences its top 10 orders correctly.

| Comparison | Δ ppref | p | Holm p |
|---|---|---|---|
| `harrier` vs `e5` | +0.295 | <0.001 | 0.001 |
| `qe-hybrid-neutral` vs `hybrid` | +0.062 | 0.001 | 0.010 |
| `qe-hybrid-vl` vs `qe-hybrid-neutral` | +0.059 | 0.002 | 0.022 |
| `hybrid-logs` vs `hybrid` | +0.040 | 0.03 | 0.27 |
| `hybrid-vl` vs `hybrid` | +0.045 | 0.04 | 0.35 |
| `harrier-ft` vs `harrier` | +0.013 | 0.74 | 1.00 |
| `qe-hybrid-rerank` vs `qe-hybrid` | +0.009 | 0.81 | 1.00 |
| `hybrid` vs `harrier` | −0.013 | 0.67 | 1.00 |

The system ordering by ppref agrees with the judge's ordering (Kendall τ = 0.70
over the 16 systems whose top 5 is mostly labelled).

**The judge against the duels.** In 551 duels the annotator picked a side. The
judge had graded the two listings differently in 340 of them, and its order
matched the annotator's pick in 75% of those. In the other 211 (38%), the
judge's grades could not tell the two listings apart.

**Listing pairs, chosen to probe the judge.** The annotator picked the better of
two listings in 45 pairs, 15 of each kind. The judge was asked the same
question in both orders.

| Pair kind | Annotator picked a side | Pick agrees with own grades | Pick agrees with judge grades | Judge's direct pick = annotator's |
|---|---|---|---|---|
| Equal annotator grades | 13/15 | n/a | 7/13 | 8/15 |
| Annotator and judge grades disagree | 12/15 | 9/12 | 3/12 | 3/15 |
| Clearly different grades | 14/15 | 14/14 | 13/14 | 14/15 |

On pairs where the annotator and the judge graded in opposite directions, the
annotator's direct choice sides with their own grades 9 times out of 12. The
judge, asked directly, repeats its own view: it matches the annotator on only 3
of 15. The judge's disagreements are therefore systematic, a bias that more
judging would not average out.

**Rankings with a divider.** For each of the 20 queries, the annotator ordered
8 listings and placed a divider where results stop being acceptable. In
context, the divider agrees with the annotator's own grade ≥ 2 at κ = 0.70. It
agrees with the judge's at κ = 0.39, the same boundary weakness the retest
showed.

**The judge side by side.** The judge was shown two top-5 lists in both
orders. It prefers `qe-hybrid` over `hybrid` 36 to 13 (47 ties), and over
`harrier` 44 to 18 (37 ties); both have sign-test p = 0.001. Only 114 of 195
verdicts survive swapping the order of the lists. List-level judging therefore
has a strong position bias, which is why the judge grades listings one at a
time.

### Historical search behaviour

`itemProfile.search` records which search terms led users to each listing. A
hashed 20% of the terms is held out: no system ever sees it. That split gives
relevance labels no LLM touched, and they describe a different query
population than the evaluation set ([`evaluation/behavioral_graded.py`](../evaluation/behavioral_graded.py)).

How the labels are built:

- Only searches the user typed count: not suggestion clicks, history, or trends.
- A listing's gain is log2(1 + the number of times users reached it from the term).
- Each term is classified by the local Gemma as navigational, generic, or descriptive. The LLM only chooses the subset; the labels stay behavioural.
- Systems that call the LLM run on the descriptive subset only.

| System | nDCG@10, all 1,566 terms | navigational (951) | generic (439) | descriptive (176) |
|---|---|---|---|---|
| `bm25-pt` | 0.086 | 0.095 | 0.065 | 0.086 |
| `e5` | 0.083 | 0.093 | 0.059 | 0.088 |
| `harrier` | 0.088 | 0.095 | 0.065 | 0.111 |
| `hybrid` | 0.097 | 0.108 | 0.070 | 0.106 |
| `hybrid-vl` | 0.100 | 0.108 | 0.074 | 0.117 |
| `harrier-ft` | 0.109 | 0.120 | 0.073 | **0.135** |
| `hybrid-logs` | **0.192** | **0.229** | **0.115** | **0.186** |
| `qe-hybrid-neutral` | — | — | — | 0.110 |
| `qe-hybrid-vl` | — | — | — | 0.116 |

On real searches, past behaviour dominates.

- **Logs.** Writing past search terms into the listing nearly doubles nDCG@10 (+0.095, p < 0.001).
- **The fine-tuned tower** adds +0.020 (p < 0.001), and +0.024 on descriptive searches (p = 0.01).
- **Fusion** helps brand searches (+0.013, p < 0.001).
- **Photos** change nothing.
- **The LLM rewrite.** On a random sample of 300 searches, it loses 0.018 MRR@10 against `hybrid` (p = 0.02), because rewriting a brand name into a dish description drifts away from it. On descriptive searches it is neutral (+0.003, p = 0.74).

The system ordering agrees only weakly with the judge's. That reflects the two
query populations more than the judge, and it is the evidence behind routing.

### The judge with the listing photo

The same Qwen3.5-9B has a vision encoder. It re-graded every human-labelled pair
with the listing photo attached; the prompt differs in one line
([`evaluation/judge_mm.py`](../evaluation/judge_mm.py)).

- **The photo makes the judge less strict.** Its mean grade rises from 1.01 to 1.14, close to the annotator's 1.16.
- **It does not make the judge agree better with the annotator.**
  - Weighted κ changes by −0.04 [95% CI −0.12, +0.02].
  - κ on acceptable changes by +0.05 [−0.06, +0.15].

The text-only judge stays primary.

## Limitations

- **One annotator,** a non-native reader working from machine translations.
- **The duels compare ranks 1–2 only.** Rank-3 duels exist for only a few queries.
- **The photo experiments share a model family with the judge.**
- **The routing recommendation is not evaluated end to end.** Nearly all 100 evaluation queries are descriptive.
- **A production evaluation set would need roughly 150–200 queries,** more than one annotator, and a held-out split decided before design.
