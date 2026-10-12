"""Generative retrieval quality by training epoch: decode saved checkpoints, judge them, summarise.

    # on the GPU machine, after `train --tag=-50ep --save-every 5`:
    python -m retrieval.generative.curve decode --variants tu-clean semantic semantic-mm --tag=-50ep
    # on the laptop, after pulling output/genret/curve-runs-*.jsonl:
    python -m retrieval.generative.curve judge
    python -m retrieval.generative.curve summary

Curve runs stay out of output/eval/runs.jsonl, so they neither enter the main judging pool nor
shift the main table. Their unjudged listings are judged with the same blind prompt and model.
nDCG@5 here divides by an ideal from every judged listing for the query (main pool and curve).
"""

from __future__ import annotations

import argparse
import json
import re
import statistics
from concurrent.futures import ThreadPoolExecutor

from core.llm import LLM
from core.local_models import pick_device
from retrieval.pipeline import item_card
from core.common import JUDGMENTS, POOL_DEPTH, ROOT, load_corpus, read_jsonl, write_jsonl
from evaluation.judge import JUDGE_MODEL, JUDGE_SCHEMA, JUDGE_SYSTEM, judge_prompt
from evaluation.metrics import bootstrap_ci, ndcg_at_k

from .ids import IDS, VARIANTS
from .run import decode_queries
from .train import CHECKPOINTS, load_model

OUT = ROOT / "output" / "genret"
def curve_runs() -> list[dict]:
    """Decoded runs of every variant (one file per variant, so parallel GPU jobs never share a file)."""
    return [record for path in sorted(OUT.glob("curve-runs-*.jsonl")) for record in read_jsonl(path)]


CURVE_JUDGMENTS = OUT / "curve-judgments.jsonl"
CURVE = OUT / "curve.json"


def decode(variants: list[str], tag: str) -> None:
    device = pick_device()
    corpus = load_corpus()
    ids = json.loads(IDS.read_text(encoding="utf-8"))
    for variant in variants:
        path_out = OUT / f"curve-runs-{variant}{tag}.jsonl"
        records = read_jsonl(path_out)
        done = {(r["system"], r["epoch"]) for r in records}
        snapshots = sorted(CHECKPOINTS.glob(f"{variant}{tag}-e*"), key=lambda path: int(re.search(r"-e(\d+)$", path.name).group(1)))
        for path in snapshots:
            epoch = int(re.search(r"-e(\d+)$", path.name).group(1))
            system = f"gr-{variant}{tag}"
            if (system, epoch) in done:
                continue
            tokenizer, model = load_model(device, path)
            model.eval()
            decoded, timings = decode_queries(model, tokenizer, ids, variant, corpus, system, device)
            records += [{**record, "epoch": epoch} for record in decoded]
            write_jsonl(path_out, records)
            print(f"{system} epoch {epoch}: median {statistics.median(timings):.0f} ms per query", flush=True)
            del model


def judge(workers: int = 8) -> None:
    corpus = load_corpus()
    index_of = {d: i for i, d in enumerate(corpus.document_ids)}
    queries = {q["id"]: q["text"] for q in corpus.queries}
    known = {(r["query_id"], r["document_id"]) for r in read_jsonl(JUDGMENTS)}
    records = read_jsonl(CURVE_JUDGMENTS)
    known |= {(r["query_id"], r["document_id"]) for r in records}
    pairs = sorted({(r["query_id"], d) for r in curve_runs() for d in r["document_ids"][:POOL_DEPTH]} - known)
    llm = LLM()

    def grade(pair):
        q, d = pair
        verdict = llm.json("judge", JUDGE_MODEL, JUDGE_SYSTEM, judge_prompt(queries[q], item_card(corpus, index_of[d])), JUDGE_SCHEMA)
        return {"query_id": q, "document_id": d, "judge": JUDGE_MODEL, **verdict}

    print(f"Judging {len(pairs)} new (query, listing) pairs", flush=True)
    with ThreadPoolExecutor(workers) as executor:
        records += list(executor.map(grade, pairs))
    write_jsonl(CURVE_JUDGMENTS, records)


def summary() -> dict:
    grades = {(r["query_id"], r["document_id"]): r["grade"] for r in read_jsonl(JUDGMENTS) + read_jsonl(CURVE_JUDGMENTS)}
    pool: dict[int, list[int]] = {}
    for (q, _), g in grades.items():
        pool.setdefault(q, []).append(g)
    result: dict = {}
    for record in curve_runs():
        ranked = [grades.get((record["query_id"], d), 0) for d in record["document_ids"][:POOL_DEPTH]]
        result.setdefault(record["system"], {}).setdefault(record["epoch"], []).append(ndcg_at_k(ranked, pool.get(record["query_id"], [])))
    summary = {}
    for system, by_epoch in result.items():
        variant = re.match(r"gr-(.+?)(-50ep)?$", system).group(1)
        history = {}
        train_file = OUT / f"train-{variant}{system[len('gr-') + len(variant):]}.json"
        if train_file.is_file():
            history = {h["epoch"]: h for h in json.loads(train_file.read_text())["history"]}
        summary[system] = {str(epoch): {"nDCG@5": statistics.fmean(values), "ci95": bootstrap_ci(values),
                                        "dev_recall@10": history.get(epoch, {}).get("recall@10"),
                                        "train_loss": history.get(epoch, {}).get("loss")}
                           for epoch, values in sorted(by_epoch.items())}
        for epoch, row in summary[system].items():
            print(f"{system:22} epoch {epoch:>3}  nDCG@5 {row['nDCG@5']:.3f} [{row['ci95'][0]:.2f}, {row['ci95'][1]:.2f}]"
                  f"  dev R@10 {row['dev_recall@10'] if row['dev_recall@10'] is not None else float('nan'):.3f}")
    CURVE.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=["decode", "judge", "summary"])
    parser.add_argument("--variants", nargs="+", choices=VARIANTS, default=["tu-clean", "semantic", "semantic-mm"])
    parser.add_argument("--tag", default="-50ep")
    args = parser.parse_args()
    if args.command == "decode":
        decode(args.variants, args.tag)
    elif args.command == "judge":
        judge()
    summary()


if __name__ == "__main__":
    main()
