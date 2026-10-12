"""Industry baseline: a two-tower retriever fine-tuned on search logs with in-batch-negative InfoNCE.

    # GPU machine
    python -m retrieval.two_tower train --data logs              # -> .demo-cache/models/harrier-ft-logs
    python -m retrieval.two_tower train --data logs+synthetic    # -> .demo-cache/models/harrier-ft-syn
    # laptop, after copying the two model folders back
    python -m retrieval.two_tower evaluate

Production food search (Uber Eats, DoorDash) trains its two-tower retriever on its own engagement
logs with a contrastive loss over in-batch negatives. Here both towers share harrier-oss-v1-0.6b,
so the result compares directly with `harrier` and still runs on the laptop. Training pairs are
(search term, listing text) from the 80% training split of itemProfile.search terms; the hashed 20%
held-out terms never enter training and remain the behavioural test set. 5% of the training terms
are kept aside to pick the epoch.

    harrier-ft       fine-tuned on search logs only
    harrier-ft-syn   search logs + the three LLM-written queries per item (synthetic, doc2query-style)
    hybrid-ft        RRF(bm25-pt, harrier-ft): the production-style hybrid with the fine-tuned tower
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import random
import time
from collections import Counter, defaultdict

import numpy as np

from core.local_models import DenseEncoder, pick_device
from retrieval.pipeline import CANDIDATE_DEPTH, DENSE_INSTRUCTION

from evaluation.behavioral import RESULT as BEHAVIOURAL, evaluate_terms, held_out, held_out_terms, paired
from core.common import ROOT, RUNS, load_corpus, load_pipeline, read_jsonl, write_jsonl

BASE = ROOT / ".demo-cache" / "models" / "harrier-oss-v1-0.6b"
MODELS = {"logs": ROOT / ".demo-cache" / "models" / "harrier-ft-logs", "logs+synthetic": ROOT / ".demo-cache" / "models" / "harrier-ft-syn"}
FT_SYSTEMS = ["harrier-ft", "harrier-ft-syn", "hybrid-ft"]
QUERY_PROMPT = f"Instruct: {DENSE_INSTRUCTION}\nQuery: "  # the same prompt DenseEncoder uses at inference
MAX_LISTINGS_PER_TERM = 20  # navigational terms ("burger king") keep their most-searched listings only


def _dev_term(term: str) -> bool:
    return int(hashlib.sha256(f"dev:{term}".encode()).hexdigest(), 16) % 20 == 0


def training_pairs(corpus, data: str) -> tuple[list[tuple[str, int]], dict[str, set[int]]]:
    """(query, content row) training pairs, and dev term -> relevant rows. Rows are first rows of distinct contents."""
    first_row: dict[str, int] = {}
    for row, text in enumerate(corpus.texts):
        first_row.setdefault(text, row)
    searches: dict[str, Counter] = defaultdict(Counter)
    for row, item in enumerate(corpus.items):
        for entry in corpus.get_item(item["document_id"])["item_profile"].get("search") or []:
            term = str(entry.get("term") or "").strip().lower()
            if term and not held_out(term):
                searches[term][first_row[corpus.texts[row]]] += int(entry.get("count") or 1)
    pairs, dev = [], {}
    for term, counter in searches.items():
        rows = [row for row, _ in counter.most_common(MAX_LISTINGS_PER_TERM)]
        if _dev_term(term):
            dev[term] = set(rows)
        else:
            pairs += [(term, row) for row in rows]
    if data == "logs+synthetic":
        for record in read_jsonl(ROOT / "output" / "genret" / "items.jsonl"):
            pairs += [(query, first_row[corpus.texts[record["row"]]]) for query in record.get("queries") or []]
    return list(dict.fromkeys(pairs)), dev


def batches(pairs: list[tuple[str, int]], size: int, seed: int) -> list[list[tuple[str, int]]]:
    """Shuffled batches without a repeated query or listing, so no in-batch negative is a false negative by identity."""
    order = pairs[:]
    random.Random(seed).shuffle(order)
    result = []
    while order:
        batch, queries, rows, rest = [], set(), set(), []
        for pair in order:
            if len(batch) < size and pair[0] not in queries and pair[1] not in rows:
                batch.append(pair)
                queries.add(pair[0])
                rows.add(pair[1])
            else:
                rest.append(pair)
        result.append(batch)
        order = rest
    return result


def train(data: str, epochs: int, batch_size: int, lr: float, scale: float) -> None:
    import torch
    from sentence_transformers import SentenceTransformer

    corpus = load_corpus()
    pairs, dev = training_pairs(corpus, data)
    device = pick_device()
    model = SentenceTransformer(str(BASE), device=device, local_files_only=True)
    model.max_seq_length = 256
    model[0].auto_model.gradient_checkpointing_enable()  # batch-128 InfoNCE on 256-token listings otherwise needs >90 GB
    texts = corpus.texts
    first_row: dict[str, int] = {}
    for row, text in enumerate(texts):
        first_row.setdefault(text, row)
    canonical = [first_row[text] for text in texts]
    print(f"{data}: {len(pairs)} training pairs, {len(dev)} dev terms, device {device}", flush=True)

    def embed(sentences: list[str]):
        features = model.tokenize(sentences)
        features = {key: value.to(device) if hasattr(value, 'to') else value for key, value in features.items()}
        with torch.autocast(device_type=device, dtype=torch.bfloat16, enabled=device == "cuda"):
            output = model(features)["sentence_embedding"]
        return torch.nn.functional.normalize(output.float(), dim=-1)

    def dev_recall() -> float:
        model.eval()
        with torch.no_grad():
            documents = model.encode(texts, batch_size=128, normalize_embeddings=True, convert_to_tensor=True)
            terms = sorted(dev)
            queries = model.encode(terms, prompt=QUERY_PROMPT, batch_size=128, normalize_embeddings=True, convert_to_tensor=True)
            top = (queries @ documents.T).topk(50, dim=1).indices.cpu().tolist()
        model.train()
        hits = 0
        for term, ranked in zip(terms, top):
            rows = list(dict.fromkeys(canonical[r] for r in ranked))[:10]  # distinct contents
            hits += any(row in dev[term] for row in rows)
        return hits / len(terms)

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01)
    steps_per_epoch = math.ceil(len(pairs) / batch_size)
    total, warmup = steps_per_epoch * epochs, max(1, steps_per_epoch * epochs // 10)
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda step: min((step + 1) / warmup, max(0.0, (total - step) / (total - warmup))))
    history = [{"epoch": 0, "dev_recall@10": dev_recall()}]
    print(json.dumps(history[-1]), flush=True)
    best, out = history[0]["dev_recall@10"], MODELS[data]
    model.train()
    for epoch in range(1, epochs + 1):
        started, losses = time.perf_counter(), []
        for batch in batches(pairs, batch_size, seed=epoch):
            queries = embed([QUERY_PROMPT + query for query, _ in batch])
            documents = embed([texts[row] for _, row in batch])
            logits = scale * queries @ documents.T  # InfoNCE: every other listing in the batch is a negative
            loss = torch.nn.functional.cross_entropy(logits, torch.arange(len(batch), device=device))
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad(set_to_none=True)
            losses.append(float(loss))
        history.append({"epoch": epoch, "loss": float(np.mean(losses)), "minutes": (time.perf_counter() - started) / 60,
                        "dev_recall@10": dev_recall()})
        print(json.dumps(history[-1]), flush=True)
        if history[-1]["dev_recall@10"] > best:
            best = history[-1]["dev_recall@10"]
            model.save(str(out))
            (out / "prosus-model.json").write_text(json.dumps({"repo_id": f"local/{out.name}", "revision": f"epoch{epoch}",
                                                               "base": "microsoft/harrier-oss-v1-0.6b"}, indent=2) + "\n")
    (ROOT / "output" / "eval" / f"two-tower-{out.name}.json").write_text(json.dumps(
        {"data": data, "pairs": len(pairs), "dev_terms": len(dev), "epochs": epochs, "batch_size": batch_size, "lr": lr,
         "scale": scale, "history": history, "best_dev_recall@10": best}, indent=2) + "\n")


def retrievers(pipeline) -> dict:
    """system -> function(query) -> top-10 row indices, for the fine-tuned towers and their hybrid."""
    engine = pipeline.engine
    encoders = {system: DenseEncoder(path, ROOT / ".demo-cache" / "retrieval", DENSE_INSTRUCTION)
                for system, path in (("harrier-ft", MODELS["logs"]), ("harrier-ft-syn", MODELS["logs+synthetic"]))}
    matrices = {system: encoder.documents(pipeline.corpus.texts) for system, encoder in encoders.items()}
    _ = pipeline.dense_matrix

    def ranking(system: str, query: str, top_k: int = 10) -> list[int]:
        if system == "hybrid-ft":
            fused, _ = pipeline._fuse([engine.rank_distinct(engine.bm25_scores(query, stem=True), CANDIDATE_DEPTH, positive_only=True),
                                       engine.rank_distinct(matrices["harrier-ft"] @ encoders["harrier-ft"].queries([query])[0], CANDIDATE_DEPTH)])
            return fused[:top_k]
        return engine.rank_distinct(matrices[system] @ encoders[system].queries([query])[0], top_k)

    ranking.encoders = encoders  # for batch-encoding many queries up front
    return {system: (lambda query, s=system: ranking(s, query)) for system in FT_SYSTEMS} | {"_ranking": ranking}


def evaluate() -> None:
    pipeline = load_pipeline()
    corpus = pipeline.corpus
    found = retrievers(pipeline)
    ranking, encoders = found["_ranking"], found["_ranking"].encoders

    records = []
    for system in FT_SYSTEMS:
        for query in corpus.queries:
            start = time.perf_counter()
            ranked = ranking(system, query["text"])
            records.append({"query_id": query["id"], "query": query["text"], "system": system,
                            "document_ids": [corpus.document_ids[i] for i in ranked],
                            "elapsed_ms": round((time.perf_counter() - start) * 1000, 1)})
    write_jsonl(RUNS, [r for r in read_jsonl(RUNS) if r["system"] not in FT_SYSTEMS] + records)
    print(f"Saved {len(records)} runs to {RUNS}", flush=True)

    terms = held_out_terms(corpus)
    names = sorted(terms)
    for encoder in encoders.values():
        encoder.queries(names)  # batch-encode the held-out terms once
    result = json.loads(BEHAVIOURAL.read_text()) if BEHAVIOURAL.is_file() else {"systems": {}}
    comparisons = {}
    for system, baseline in (("harrier-ft", "harrier"), ("harrier-ft-syn", "harrier"), ("hybrid-ft", "hybrid")):
        result["systems"][system] = evaluate_terms(corpus, terms, names, lambda term, s=system: ranking(s, term))
        if comparison := paired(result, system, baseline):
            comparisons[system] = comparison
        print(f"{system:15} held-out terms {len(names)}  MRR@10 {result['systems'][system]['mrr@10']:.3f}"
              + (f"  vs {baseline} {comparison['baseline_mrr@10']:.3f} (p={comparison['p_value']:.4f})" if comparison else ""), flush=True)
    result["two_tower_comparisons"] = comparisons
    BEHAVIOURAL.write_text(json.dumps(result, indent=1, ensure_ascii=False), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=["train", "evaluate"])
    parser.add_argument("--data", choices=sorted(MODELS), default="logs")
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--lr", type=float, default=2e-5)
    parser.add_argument("--scale", type=float, default=20.0, help="inverse temperature of the InfoNCE logits")
    args = parser.parse_args()
    if args.command == "train":
        train(args.data, args.epochs, args.batch_size, args.lr, args.scale)
    else:
        evaluate()


if __name__ == "__main__":
    main()
