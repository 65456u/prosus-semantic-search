"""Evaluate every system with the human listwise rankings and listing-pair preferences only (no grades).

Listing-pair preferences come from the first listing-pair round and from the same-rank duels
(tools.annotation.pairwise prepare-duels).

    .venv/bin/python -m evaluation.preference_eval

For each labelled query the annotator ordered 8 listings and placed a divider where results stop
being acceptable. Measures, following the preference-judgment literature:

    compat   compatibility (Clarke, Vtyurina & Smucker, TOIS 2021): RBO (p = 0.85) between the
             system's ranking and the ideal ranking (the listings above the divider, in the
             annotator's order), normalised so the ideal ranking scores 1
    nDCG@5   nDCG with gains from the listwise order: the k-th listing above the divider gets
             gain (cut - k), listings below it 0
    acc@5    share of the system's top labelled listings that lie above the divider
    ppref    share of human preferences the system orders correctly (Carterette & Bennett, SIGIR
             2008): every pair implied by a listwise order plus the listing-pair choices; a
             retrieved listing beats one not retrieved; pairs with neither listing retrieved are skipped
    wpref    ppref with each pair weighted by 1 / log2(1 + rank of its higher-ranked listing)

Only 8 listings per query are labelled, so compat, nDCG@5 and acc@5 use condensed lists: the
system's top 10 with unlabelled listings removed (Sakai, SIGIR 2007). Coverage@5 reports how many
of the system's top 5 listings are labelled; condensed scores of low-coverage systems rest on few
listings. Writes output/eval/preference_eval.json and .md.
"""

from __future__ import annotations

import json
import math
import statistics
from datetime import datetime

from core.common import OUT, RUNS, read_jsonl
from evaluation.metrics import bootstrap_ci, kendall_tau, paired_permutation_p

P = 0.85  # RBO persistence Clarke et al. use for a top-5 ideal ranking
DEPTH = 10
RESULT = OUT / "preference_eval.json"
COMPARISONS = [("harrier", "e5"), ("hybrid", "harrier"), ("qe-hybrid", "hybrid"), ("qe-hybrid-neutral", "hybrid"),
               ("qe-hybrid-rerank", "qe-hybrid"), ("hybrid-vl", "hybrid"), ("hybrid-vl", "qe-hybrid-neutral"),
               ("qe-hybrid-vl", "qe-hybrid-neutral"), ("vl-fused", "harrier"), ("harrier-ft", "harrier"),
               ("hybrid-ft", "hybrid"), ("hybrid-logs", "hybrid")]


def rbo(ranking: list[str], ideal: list[str], p: float = P) -> float:
    depth = max(len(ranking), len(ideal))
    return (1 - p) * sum(p ** (i - 1) * len(set(ranking[:i]) & set(ideal[:i])) / i for i in range(1, depth + 1))


def labels() -> tuple[dict[int, dict], dict[int, list[tuple[str, str]]]]:
    """Latest listwise record per query, and per query the preferences (winner, loser) from both sources."""
    listwise = {r["query_id"]: r for r in read_jsonl(OUT / "human_listwise.jsonl")}
    prefs: dict[int, list[tuple[str, str]]] = {}
    for q, record in listwise.items():
        order = record["order"]
        prefs.setdefault(q, []).extend((a, b) for i, a in enumerate(order) for b in order[i + 1:])
    pair_records = read_jsonl(OUT / "human_item_pairwise.jsonl") + read_jsonl(OUT / "human_duel_pairwise.jsonl")
    for record in {r["id"]: r for r in pair_records}.values():
        if record["winner"] != "tie":
            loser = record["right_doc"] if record["winner"] == record["left_doc"] else record["left_doc"]
            prefs.setdefault(record["query_id"], []).append((record["winner"], loser))
    return listwise, prefs


def per_query(ranking: list[str], record: dict | None, pairs: list[tuple[str, str]]) -> dict[str, float | None]:
    out: dict[str, float | None] = {"compat": None, "nDCG@5": None, "acc@5": None, "ppref": None, "wpref": None, "coverage@5": None}
    if record:
        labelled, cut = record["order"], record["cut"]
        ideal = labelled[:cut]
        condensed = [d for d in ranking[:DEPTH] if d in labelled]
        out["coverage@5"] = sum(d in labelled for d in ranking[:5]) / 5
        if cut and condensed:
            out["compat"] = rbo(condensed, ideal) / rbo(ideal, ideal)
            gain = {d: cut - i for i, d in enumerate(ideal)}
            dcg = sum(gain.get(d, 0) / math.log2(i + 2) for i, d in enumerate(condensed[:5]))
            idcg = sum(g / math.log2(i + 2) for i, g in enumerate(sorted(gain.values(), reverse=True)[:5]))
            out["nDCG@5"] = dcg / idcg
            out["acc@5"] = sum(d in ideal for d in condensed[:5]) / len(condensed[:5])
    rank = {d: i + 1 for i, d in enumerate(ranking[:DEPTH])}
    right = wrong = w_right = w_wrong = 0.0
    for winner, loser in pairs:
        a, b = rank.get(winner), rank.get(loser)
        if a is None and b is None:
            continue
        weight = 1 / math.log2(1 + min(r for r in (a, b) if r is not None))
        if a is not None and (b is None or a < b):
            right, w_right = right + 1, w_right + weight
        else:
            wrong, w_wrong = wrong + 1, w_wrong + weight
    if right + wrong:
        out["ppref"], out["wpref"] = right / (right + wrong), w_right / (w_right + w_wrong)
    return out


def main() -> None:
    listwise, prefs = labels()
    runs = {(r["system"], r["query_id"]): r["document_ids"] for r in read_jsonl(RUNS)}
    systems = sorted({s for s, _ in runs}, key=lambda s: list(dict.fromkeys(s for s, _ in runs)).index(s))
    queries = sorted(set(listwise) | set(prefs))
    scores = {s: {q: per_query(runs[(s, q)], listwise.get(q), prefs.get(q, [])) for q in queries} for s in systems}
    metrics = ["compat", "nDCG@5", "acc@5", "ppref", "wpref", "coverage@5"]
    summary = {}
    for s in systems:
        summary[s] = {}
        for m in metrics:
            values = [v[m] for v in scores[s].values() if v[m] is not None]
            summary[s][m] = {"mean": statistics.fmean(values) if values else None, "n": len(values),
                             "ci95": bootstrap_ci(values) if len(values) > 1 else None}

    def paired(a: str, b: str, metric: str) -> dict | None:
        both = [q for q in queries if scores[a][q][metric] is not None and scores[b][q][metric] is not None]
        if len(both) < 5:
            return None
        x, y = [scores[a][q][metric] for q in both], [scores[b][q][metric] for q in both]
        d = [i - j for i, j in zip(x, y)]
        sd = statistics.stdev(d) if len(d) > 1 else 0.0
        return {"n": len(both), "delta": statistics.fmean(d), "ci95": bootstrap_ci(d), "p": paired_permutation_p(x, y), "sd": sd,
                # queries needed for 80% power at alpha 0.05 to detect a difference of 0.05 / 0.10
                "queries_for_0.05": math.ceil((2.8 * sd / 0.05) ** 2) if sd else None,
                "queries_for_0.10": math.ceil((2.8 * sd / 0.10) ** 2) if sd else None}

    comparisons = {f"{a} vs {b}": {m: paired(a, b, m) for m in ("compat", "ppref")} for a, b in COMPARISONS if a in scores and b in scores}

    # Consistency with the graded evaluations: system orderings by compat vs judge nDCG and human pointwise nDCG.
    report = json.loads((OUT / "report.json").read_text())
    judge = {s: v["nDCG@5"]["mean"] for s, v in report["llm_judge"].items()}
    human = report["human_validation"].get("system_nDCG_human", {})
    covered = [s for s in systems if (summary[s]["coverage@5"]["mean"] or 0) >= 0.4 and summary[s]["compat"]["mean"] is not None]
    consistency = {
        "systems_with_coverage_0.4": covered,
        "tau_compat_vs_judge": kendall_tau([summary[s]["compat"]["mean"] for s in covered], [judge[s] for s in covered]) if len(covered) > 2 else None,
        "tau_compat_vs_human_pointwise": (kendall_tau([summary[s]["compat"]["mean"] for s in covered if s in human], [human[s] for s in covered if s in human])
                                          if sum(s in human for s in covered) > 2 else None),
        "tau_ppref_vs_judge": kendall_tau([summary[s]["ppref"]["mean"] for s in covered], [judge[s] for s in covered]) if len(covered) > 2 else None,
    }

    # Annotation effort, from the label timestamps.
    def median_gap(records: list[dict]) -> float:
        times = sorted(datetime.fromisoformat(r["time"]) for r in records)
        gaps = [(b - a).total_seconds() for a, b in zip(times, times[1:]) if 0 < (b - a).total_seconds() < 600]
        return statistics.median(gaps) if gaps else float("nan")

    effort = {"listwise_seconds_per_query": median_gap(read_jsonl(OUT / "human_listwise.jsonl")),
              "pair_seconds_per_pair": median_gap(read_jsonl(OUT / "human_item_pairwise.jsonl"))}
    result = {"queries_listwise": len(listwise), "queries_with_acceptable": sum(1 for r in listwise.values() if r["cut"]),
              "preference_pairs": sum(map(len, prefs.values())), "systems": summary, "comparisons": comparisons,
              "consistency": consistency, "effort": effort}
    RESULT.write_text(json.dumps(result, indent=1), encoding="utf-8")

    def f(x, d=3):
        return "—" if x is None else f"{x:.{d}f}"

    lines = ["# Preference-based evaluation (listwise rankings + listing pairs, no grades)", "",
             f"{result['queries_listwise']} listwise queries ({result['queries_with_acceptable']} with at least one acceptable listing), "
             f"{result['preference_pairs']} human preferences. compat/nDCG@5/acc@5 use condensed lists; coverage@5 = labelled share of the top 5.", "",
             "| System | coverage@5 | compat [95% CI] | nDCG@5 | acc@5 | ppref | wpref |", "|---|---|---|---|---|---|---|"]
    for s in systems:
        v = summary[s]
        ci = v["compat"]["ci95"]
        lines.append(f"| `{s}` | {f(v['coverage@5']['mean'], 2)} | {f(v['compat']['mean'])}" + (f" [{ci[0]:.2f}, {ci[1]:.2f}]" if ci else "")
                     + f" (n={v['compat']['n']}) | {f(v['nDCG@5']['mean'])} | {f(v['acc@5']['mean'])} | {f(v['ppref']['mean'])} | {f(v['wpref']['mean'])} |")
    lines += ["", "| Comparison | Δ compat [95% CI], p (n) | Δ ppref, p | queries for 0.05 / 0.10 (compat) |", "|---|---|---|---|"]
    for name, c in comparisons.items():
        cc, cp = c["compat"], c["ppref"]
        lines.append(f"| {name} | " + (f"{cc['delta']:+.3f} [{cc['ci95'][0]:+.2f}, {cc['ci95'][1]:+.2f}], p={cc['p']:.3f} ({cc['n']})" if cc else "—")
                     + " | " + (f"{cp['delta']:+.3f}, p={cp['p']:.3f}" if cp else "—")
                     + " | " + (f"{cc['queries_for_0.05']} / {cc['queries_for_0.10']}" if cc else "—") + " |")
    lines += ["", f"Kendall τ of system orderings (systems with coverage@5 ≥ 0.4: {len(covered)}): compat vs judge nDCG "
              f"{f(consistency['tau_compat_vs_judge'], 2)}, compat vs human pointwise nDCG {f(consistency['tau_compat_vs_human_pointwise'], 2)}, "
              f"ppref vs judge {f(consistency['tau_ppref_vs_judge'], 2)}.",
              f"Effort: about {effort['listwise_seconds_per_query']:.0f} s per listwise query (ranking and divider passes counted separately), "
              f"{effort['pair_seconds_per_pair']:.0f} s per listing pair."]
    (OUT / "preference_eval.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
