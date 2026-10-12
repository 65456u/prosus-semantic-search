"""Multimodal judge check: does showing the listing photo bring the judge closer to the human?

    .venv/bin/python -m core.images --human-only     # fetch photos first
    .venv/bin/python -m evaluation.judge_mm [--workers 4]

The same judge model (Qwen3.5 9B, which has a vision encoder) re-grades every human-labelled
(query, listing) pair with the listing photo attached. The prompt is the text judge's prompt with
one line changed, so the only difference is the photo. Listings without a photo are graded from
text alone by the same prompt, which keeps the pair set identical to the text judge's.

Compared against the human on the same pairs: weighted κ, κ on acceptable (≥2), Kendall τ-b with
the unified human gold, and κ with the listwise divider (in-context acceptability).
"""

from __future__ import annotations

import argparse
import base64
import json
import statistics
from concurrent.futures import ThreadPoolExecutor

from core.llm import LLM
from retrieval.pipeline import item_card

from core.common import HUMAN, JUDGMENTS, OUT, load_corpus, read_jsonl, write_jsonl
from evaluation.gold import divider_labels, unified_gold
from core.images import image_path
from evaluation.judge import JUDGE_MODEL, JUDGE_SCHEMA, JUDGE_SYSTEM, judge_prompt
from evaluation.metrics import cohen_kappa, kendall_tau_b

JUDGMENTS_MM = OUT / "judgments_mm.jsonl"
TEXT_ONLY_LINE = "Judge only from the listing text. Do not reward a listing for sharing words with the query when the food differs."
assert TEXT_ONLY_LINE in JUDGE_SYSTEM
JUDGE_SYSTEM_MM = JUDGE_SYSTEM.replace(
    TEXT_ONLY_LINE,
    "Judge from the listing text and, when attached, the listing photo: the photo shows what is actually served "
    "(the dish, its style, portion, or packaging). Do not reward a listing for sharing words with the query when the food differs.")


def photo(item: dict) -> str | None:
    if not item["image_url"]:
        return None
    path = image_path(item["image_url"])
    return base64.b64encode(path.read_bytes()).decode() if path.is_file() and path.stat().st_size else None


def agreement(human: list[int], judge: list[int]) -> dict[str, float]:
    return {"pairs": len(human), "exact": sum(a == b for a, b in zip(human, judge)) / len(human),
            "weighted_kappa": cohen_kappa(human, judge, [0, 1, 2, 3], "quadratic"),
            "kappa_acceptable": cohen_kappa([int(g >= 2) for g in human], [int(g >= 2) for g in judge], [0, 1])}


def summary() -> dict:
    human = {(r["query_id"], r["document_id"]): r["grade"] for r in read_jsonl(HUMAN)}
    text = {(r["query_id"], r["document_id"]): r["grade"] for r in read_jsonl(JUDGMENTS)}
    mm = {(r["query_id"], r["document_id"]): r for r in read_jsonl(JUDGMENTS_MM)}
    keys = [k for k, g in human.items() if g is not None and k in text and k in mm]
    result = {}
    for name, subset in (("all", keys), ("with_photo", [k for k in keys if mm[k]["has_photo"]])):
        h = [human[k] for k in subset]
        result[name] = {"text_judge": agreement(h, [text[k] for k in subset]),
                        "multimodal_judge": agreement(h, [mm[k]["grade"] for k in subset])}
        # Per pair: did the photo move the judge toward or away from the human grade?
        moved = [(abs(mm[k]["grade"] - human[k]), abs(text[k] - human[k])) for k in subset]
        result[name]["closer"] = sum(a < b for a, b in moved)
        result[name]["further"] = sum(a > b for a, b in moved)
    gold = unified_gold()
    taus = {"text_judge": [], "multimodal_judge": []}
    for q, strengths in gold.items():
        docs = [d for d in strengths if (q, d) in text and (q, d) in mm]
        for name, grades in (("text_judge", [text[(q, d)] for d in docs]), ("multimodal_judge", [mm[(q, d)]["grade"] for d in docs])):
            if len(set(grades)) > 1:
                taus[name].append(kendall_tau_b([strengths[d] for d in docs], grades))
    result["unified_gold_tau_b"] = {name: statistics.fmean(values) for name, values in taus.items() if values}
    divider = [(label, key) for key, label in divider_labels().items() if key in text and key in mm]
    if divider:
        result["divider_kappa"] = {
            "pairs": len(divider),
            "text_judge": cohen_kappa([l for l, _ in divider], [int(text[k] >= 2) for _, k in divider], [0, 1]),
            "multimodal_judge": cohen_kappa([l for l, _ in divider], [int(mm[k]["grade"] >= 2) for _, k in divider], [0, 1])}
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", nargs="?", choices=["judge", "summary"], default="judge")
    parser.add_argument("--judge-model", default=JUDGE_MODEL)
    parser.add_argument("--workers", type=int, default=4)
    args = parser.parse_args()
    if args.command == "judge":
        corpus = load_corpus()
        index_of = {d: i for i, d in enumerate(corpus.document_ids)}
        item_by_id = {item["document_id"]: item for item in corpus.items}
        queries = {q["id"]: q["text"] for q in corpus.queries}
        # Every listing a human graded or ranked (pointwise, listwise, or in a listing pair).
        keys = {(r["query_id"], r["document_id"]) for r in read_jsonl(HUMAN)}
        keys |= {(r["query_id"], d) for r in read_jsonl(OUT / "human_listwise.jsonl") for d in r["order"]}
        keys = sorted(keys)
        llm = LLM()

        def judge(key):
            query_id, document_id = key
            image = photo(item_by_id[document_id])
            verdict = llm.json("judge-multimodal", args.judge_model, JUDGE_SYSTEM_MM,
                               judge_prompt(queries[query_id], item_card(corpus, index_of[document_id])), JUDGE_SCHEMA,
                               images=[image] if image else None)
            return {"query_id": query_id, "document_id": document_id, "judge": args.judge_model,
                    "has_photo": image is not None, **verdict}

        print(f"Judging {len(keys)} pairs with photos", flush=True)
        with ThreadPoolExecutor(args.workers) as executor:
            records = list(executor.map(judge, keys))
        write_jsonl(JUDGMENTS_MM, records)
        print(f"Saved {len(records)} judgments ({sum(r['has_photo'] for r in records)} with a photo) to {JUDGMENTS_MM}")
    result = summary()
    (OUT / "judge_mm.json").write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
