"""Search systems built from local models, as an ablation ladder.

Each system adds a single component to the one before it, so the evaluation
measures what every component contributes:

    bm25               accent-insensitive keyword baseline (no stemming)
    bm25-pt            + Portuguese Snowball stemming
    e5                 dense baseline: multilingual-e5-small
    harrier            stronger dense retriever: harrier-oss-v1-0.6b (instruction-prompted)
    hybrid             RRF(bm25-pt, harrier)
    qe-hybrid          + LLM query understanding (multi-query RRF over raw and rewritten queries)
    qe-hybrid-rerank   + cross-encoder reranking of the top candidates against the query and its interpreted intent
    qe-hybrid-neutral  qe-hybrid with a prompt stripped of hints learned by reading the evaluation queries
                       (loanword examples, "queries may be literal translations"): a leakage check

Every system returns distinct (itemId, merchantId) listings as document row
indices. Everything runs locally; the evaluation judge never sees which system
produced a result, and it is a different model family from every system component.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from core.corpus import Corpus
from core.llm import LLM
from core.local_models import CrossEncoderReranker, DenseEncoder
from retrieval.engine import RetrievalEngine, reciprocal_rank_fusion

UNDERSTAND_MODEL = "ollama/gemma4:e4b-it-q4_K_M"
DENSE_INSTRUCTION = "Given a food delivery search query in Portuguese, retrieve catalog items that match what the user wants to order"
CANDIDATE_DEPTH = 100
RERANK_DEPTH = 30

SYSTEMS = ["bm25", "bm25-pt", "e5", "harrier", "hybrid", "qe-hybrid", "qe-hybrid-rerank", "qe-hybrid-neutral"]
LLM_SYSTEMS = {"qe-hybrid", "qe-hybrid-rerank", "qe-hybrid-neutral"}

UNDERSTAND_SYSTEM = """You help a Brazilian food-delivery search engine (iFood) understand queries.
The catalog mixes restaurant dishes with supermarket, pharmacy, and household products.
Item texts are short Brazilian Portuguese menu entries: names, optional descriptions, and menu categories.
Brazilian menus often use English loanwords (for example wrap, bowl, burger, combo) and abbreviations such as "c/" for "com".
Queries may be literal word-for-word translations of English food phrases, so first recover the food the user actually means.

For the query, return:
- interpretation_en: one sentence on what the user most likely wants.
- rewrite_pt: one natural Brazilian Portuguese sentence describing that food as a Brazilian menu would.
- dish_names_pt: 3 to 6 concrete dish or menu-item names that Brazilian menus actually use (loanwords allowed).
- ingredients_pt: 2 to 6 defining ingredients or attributes, in Brazilian Portuguese.
Do not invent brands. If the query names a non-food product, describe that product instead."""

# The same task without anything learned from the evaluation queries.
UNDERSTAND_SYSTEM_NEUTRAL = """You help a Brazilian food-delivery search engine (iFood) understand queries.
The catalog mixes restaurant dishes with supermarket, pharmacy, and household products.
Item texts are short Brazilian Portuguese menu entries: names, optional descriptions, and menu categories.

For the query, return:
- interpretation_en: one sentence on what the user most likely wants.
- rewrite_pt: one natural Brazilian Portuguese sentence describing that food as a Brazilian menu would.
- dish_names_pt: 3 to 6 concrete dish or menu-item names that Brazilian menus actually use.
- ingredients_pt: 2 to 6 defining ingredients or attributes, in Brazilian Portuguese.
Do not invent brands. If the query names a non-food product, describe that product instead."""

UNDERSTAND_SCHEMA = {
    "type": "object",
    "additionalProperties": False,
    "required": ["interpretation_en", "rewrite_pt", "dish_names_pt", "ingredients_pt"],
    "properties": {
        "interpretation_en": {"type": "string"},
        "rewrite_pt": {"type": "string"},
        "dish_names_pt": {"type": "array", "items": {"type": "string"}},
        "ingredients_pt": {"type": "array", "items": {"type": "string"}},
    },
}


def item_card(corpus: Corpus, index: int, description_chars: int = 240) -> str:
    """Compact listing text for LLM prompts: the same fields used for retrieval."""
    detail = corpus.get_item(corpus.document_ids[index])
    metadata = detail["item_metadata"]
    taxonomy = metadata.get("taxonomy") or {}
    path = " > ".join(str(taxonomy.get(level)) for level in ("l0", "l1", "l2") if taxonomy.get(level))
    description = (metadata.get("description") or "").strip()
    if len(description) > description_chars:
        description = description[:description_chars].rstrip() + "…"
    parts = [f"name: {metadata.get('name') or ''}"]
    if description:
        parts.append(f"description: {description}")
    parts.append(f"menu category: {metadata.get('category_name') or ''}")
    if path:
        parts.append(f"taxonomy: {path}")
    return " | ".join(parts)


class SearchPipeline:
    def __init__(self, engine: RetrievalEngine, llm: LLM, dense: DenseEncoder, reranker: CrossEncoderReranker,
                 understand_model: str = UNDERSTAND_MODEL):
        self.engine = engine
        self.corpus = engine.corpus
        self.llm = llm
        self.dense = dense
        self.reranker = reranker
        self.understand_model = understand_model
        self._dense_matrix: np.ndarray | None = None

    @property
    def dense_matrix(self) -> np.ndarray:
        if self._dense_matrix is None:
            self._dense_matrix = self.dense.documents(self.corpus.texts)
        return self._dense_matrix

    def dense_scores(self, text: str) -> np.ndarray:
        return self.dense_matrix @ self.dense.queries([text])[0]

    def understand(self, query: str, neutral: bool = False) -> dict[str, Any]:
        system = UNDERSTAND_SYSTEM_NEUTRAL if neutral else UNDERSTAND_SYSTEM
        return self.llm.json("query-understanding", self.understand_model, system, f"Query: {query}", UNDERSTAND_SCHEMA)

    def _fuse(self, rankings: list[list[int]]) -> tuple[list[int], dict[int, float]]:
        # Duplicate source rows of one listing must count as one RRF candidate.
        representative: dict[tuple[str, str], int] = {}
        canonical = []
        for ranking in rankings:
            canonical.append([representative.setdefault(
                (self.corpus.items[i]["item_id"], self.corpus.items[i]["merchant_id"]), i) for i in ranking])
        fused = reciprocal_rank_fusion(canonical)
        return sorted(fused, key=lambda index: (-fused[index], index)), fused

    @staticmethod
    def dense_expansion(understanding: dict[str, Any]) -> str:
        return "\n".join([understanding["rewrite_pt"], ", ".join(understanding["dish_names_pt"])])

    def expanded_rankings(self, query: str, understanding: dict[str, Any]) -> list[list[int]]:
        """The four candidate lists that query understanding fuses: keywords and harrier, each on the raw query
        and on the LLM expansion. Raw-query lists guard against an expansion that drifts from the user's words."""
        engine, depth = self.engine, CANDIDATE_DEPTH
        lexical_expansion = " ".join([query, *understanding["dish_names_pt"], *understanding["ingredients_pt"]])
        return [
            engine.rank_distinct(engine.bm25_scores(query, stem=True), depth, positive_only=True),
            engine.rank_distinct(engine.bm25_scores(lexical_expansion, stem=True), depth, positive_only=True),
            engine.rank_distinct(self.dense_scores(query), depth),
            engine.rank_distinct(self.dense_scores(self.dense_expansion(understanding)), depth),
        ]

    def rerank(self, query: str, candidates: list[int]) -> tuple[list[int], dict[int, float]]:
        logits = self.reranker.score(query, [self.corpus.texts[index] for index in candidates])
        scores = dict(zip(candidates, logits))
        # Stable sort: the fused retrieval order breaks exact ties.
        order = sorted(range(len(candidates)), key=lambda position: (-logits[position], position))
        return [candidates[position] for position in order], scores

    def run(self, system: str, query: str, top_k: int = 10) -> dict[str, Any]:
        """Ranked distinct listings for one system, with intermediate signals for analysis."""
        engine, depth = self.engine, CANDIDATE_DEPTH
        info: dict[str, Any] = {}
        if system == "bm25":
            ranking = engine.rank_distinct(engine.bm25_scores(query), top_k)
        elif system == "bm25-pt":
            ranking = engine.rank_distinct(engine.bm25_scores(query, stem=True), top_k)
        elif system == "e5":
            ranking = engine.rank_distinct(engine.e5_scores(query), top_k)
        elif system == "harrier":
            ranking = engine.rank_distinct(self.dense_scores(query), top_k)
        elif system == "hybrid":
            ranking, fused = self._fuse([engine.rank_distinct(engine.bm25_scores(query, stem=True), depth, positive_only=True),
                                  engine.rank_distinct(self.dense_scores(query), depth)])
        elif system in LLM_SYSTEMS:
            understanding = self.understand(query, neutral=system == "qe-hybrid-neutral")
            info["understanding"] = understanding
            ranking, fused = self._fuse(self.expanded_rankings(query, understanding))
            if system == "qe-hybrid-rerank":
                # Literal translations mislead a cross-encoder too ("carrinho" matches toy cars), so it
                # scores candidates against the raw query plus the interpreted intent.
                rerank_query = f"{query} ({understanding['interpretation_en']})"
                reranked, scores = self.rerank(rerank_query, ranking[:RERANK_DEPTH])
                ranking = reranked + ranking[RERANK_DEPTH:]
                info["rerank_scores"] = {self.corpus.document_ids[i]: s for i, s in scores.items()}
                info["max_rerank_score"] = max(scores.values()) if scores else None
        else:
            raise ValueError(f"Unknown system: {system}")
        if system in ("hybrid", "qe-hybrid"):
            info["fusion_scores"] = {self.corpus.document_ids[i]: fused[i] for i in ranking[:top_k]}
        return {"system": system, "query": query, "ranking": ranking[:top_k], **info}
