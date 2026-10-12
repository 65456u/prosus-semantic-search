"""Evaluation report from saved runs, LLM judgments, and human labels.

    .venv/bin/python -m evaluation.report

Writes output/eval/report.md and report.json. Everything is computed from saved
files; the only model-server call reads local Ollama model digests.
"""

from __future__ import annotations

import io
import json
import re
import statistics
from collections import Counter, defaultdict
from contextlib import redirect_stdout
from typing import Any

from core.corpus import normalize_text
from core.llm import LLM, usage_summary
from retrieval.pipeline import SYSTEMS, UNDERSTAND_MODEL
from retrieval.generative.run import GR_SYSTEMS

from core.common import HUMAN, JUDGMENTS, OUT, POOL_DEPTH, QUERY_INFO, ROOT, RUNS, load_corpus, read_jsonl
from tools.annotation.annotate import HUMAN_RETEST
from retrieval.log_expansion import LOG_SYSTEMS
from retrieval.two_tower import FT_SYSTEMS
from evaluation.gold import divider_labels, unified_gold
from evaluation.judge import ANNOTATION_MODEL
from tools.annotation.pairwise import summary as pairwise_summary, summary_items as pairwise_summary_items
from retrieval.multimodal import VL_SYSTEMS
from evaluation.metrics import (bootstrap_ci, cohen_kappa, hit_at_k, kendall_tau, kendall_tau_b, ndcg_at_k, paired_permutation_p,
                      precision_at_k, reciprocal_rank)

# Taxonomy roots that are clearly not food. Ambiguous roots (OUTROS, BEBE) are excluded from this check.
NON_FOOD = {"BELEZA_ESTETICA_HIGIENE_PESSOAL", "CASA_DECORACAO", "SAUDE_E_BEM_ESTAR", "PET", "ELETRO_E_ELETRONICOS",
            "JOGOS_BRINQUEDOS", "PAPELARIA_LIVRARIA", "JARDIM_FLORICULTURA", "MODA"}
METRICS = {"nDCG@5": lambda ranked, pool: ndcg_at_k(ranked, pool), "P@5": lambda ranked, pool: precision_at_k(ranked),
           "MRR@5": lambda ranked, pool: reciprocal_rank(ranked), "Hit@5": lambda ranked, pool: hit_at_k(ranked)}


LADDER = ["bm25", "bm25-pt", "e5", "harrier", "hybrid", "qe-hybrid", "qe-hybrid-rerank"]
# Systems outside the ablation ladder, each tested against the system it modifies.
COMPARE_WITH = {
    "qe-hybrid-neutral": "hybrid",  # leakage check: does query understanding help without eval-derived hints?
    "harrier-logs": "harrier", "hybrid-logs": "hybrid",  # past search terms written into the listing text
    "vl-text": "harrier",  # LLM-based embedder vs the smaller text embedder
    "vl-image": "vl-text", "vl-fused": "vl-text",  # photo instead of / with text, same model
    "hybrid-vl": "hybrid",  # a multimodal channel added to the hybrid
    "qe-hybrid-vl": "qe-hybrid-neutral",  # photos on top of query understanding
    "harrier-ft": "harrier", "harrier-ft-syn": "harrier-ft",  # two-tower fine-tuned on search logs (+ synthetic queries)
    "hybrid-ft": "hybrid",  # the hybrid with the fine-tuned tower
    "gr-semantic-mm": "gr-semantic",  # photo-aware semantic identifiers, same training budget
    **{f"gr-{v}-long": f"gr-{v}" for v in ("tu-raw", "tu-clean", "semantic", "semantic-mm")},  # more epochs
    **{f"gr-{v}-50ep": f"gr-{v}-long" for v in ("tu-raw", "tu-clean", "semantic", "semantic-mm")},
}


def holm(p_values: list[float]) -> list[float]:
    """Holm-Bonferroni adjusted p-values, in the input order."""
    order = sorted(range(len(p_values)), key=lambda i: p_values[i])
    adjusted, running = [0.0] * len(p_values), 0.0
    for rank, i in enumerate(order):
        running = max(running, min(1.0, (len(p_values) - rank) * p_values[i]))
        adjusted[i] = running
    return adjusted


def per_query_scores(runs: dict[tuple[str, int], list[str]], grades: dict[tuple[int, str], int],
                     systems: list[str], query_ids: list[int], ideal: dict[int, list[int]] | None = None) -> dict[str, dict[str, list[float]]]:
    pool_grades: dict[int, list[int]] = defaultdict(list)
    for (query_id, _), grade in grades.items():
        pool_grades[query_id].append(grade)
    if ideal is not None:
        pool_grades = defaultdict(list, ideal)
    scores: dict[str, dict[str, list[float]]] = {system: defaultdict(list) for system in systems}
    for system in systems:
        for query_id in query_ids:
            ranked = [grades.get((query_id, document_id), 0) for document_id in runs[(system, query_id)][:POOL_DEPTH]]
            for name, metric in METRICS.items():
                scores[system][name].append(metric(ranked, pool_grades[query_id]))
    return scores


def table(rows: list[list[Any]], header: list[str]) -> str:
    lines = ["| " + " | ".join(header) + " |", "|" + "|".join("---" for _ in header) + "|"]
    lines += ["| " + " | ".join(str(cell) for cell in row) + " |" for row in rows]
    return "\n".join(lines)


def main() -> None:
    corpus = load_corpus()
    item_by_id = {item["document_id"]: item for item in corpus.items}
    run_records = read_jsonl(RUNS)
    systems = [system for system in SYSTEMS + LOG_SYSTEMS + VL_SYSTEMS + FT_SYSTEMS + GR_SYSTEMS
               if any(record["system"] == system for record in run_records)]
    runs = {(record["system"], record["query_id"]): record["document_ids"] for record in run_records}
    judgments = read_jsonl(JUDGMENTS)
    llm_grades = {(record["query_id"], record["document_id"]): record["grade"] for record in judgments}
    info = {record["query_id"]: record for record in read_jsonl(QUERY_INFO)}
    query_ids = sorted({record["query_id"] for record in run_records})
    report: dict[str, Any] = {"queries": len(query_ids), "systems": systems, "judged_pairs": len(llm_grades)}
    md = ["# Evaluation report", "",
          f"{len(query_ids)} queries · {len(systems)} systems · {len(llm_grades)} pooled (query, listing) pairs judged "
          f"blind by `{judgments[0]['judge'] if judgments else 'n/a'}` · pool = union of every system's top {POOL_DEPTH}.", ""]

    # 1. Main LLM-judged results with bootstrap CIs and adjacent ablation tests.
    scores = per_query_scores(runs, llm_grades, systems, query_ids)
    rows, report["llm_judge"] = [], {}
    for position, system in enumerate(systems):
        entry = {}
        for name, values in scores[system].items():
            low, high = bootstrap_ci(values)
            entry[name] = {"mean": statistics.fmean(values), "ci95": [low, high]}
        nonfood = [item_by_id[d]["taxonomy_l0"] in NON_FOOD for q in query_ids for d in runs[(system, q)][:POOL_DEPTH]]
        entry["non_food@5"] = sum(nonfood) / len(nonfood)
        hub = Counter(d for q in query_ids for d in runs[(system, q)][:POOL_DEPTH]).most_common(1)[0]
        entry["most_repeated_listing"] = {"name": item_by_id[hub[0]]["name"], "queries": hub[1]}
        previous = systems[position - 1] if position else None
        if system == next((s for s in systems if s in GR_SYSTEMS), None) and "harrier" in systems:
            previous = "harrier"  # generative retrieval is compared with the best plain dense retriever
        if COMPARE_WITH.get(system) in systems:
            previous = COMPARE_WITH[system]
        if previous:
            differences = [a - b for a, b in zip(scores[system]["nDCG@5"], scores[previous]["nDCG@5"])]
            entry["vs_previous"] = {"system": previous, "delta_nDCG@5": statistics.fmean(differences),
                                    "p_value": paired_permutation_p(scores[system]["nDCG@5"], scores[previous]["nDCG@5"]),
                                    # Smallest difference a paired test detects at 80% power, alpha 0.05.
                                    "mde_80": (1.96 + 0.84) * statistics.stdev(differences) / len(differences) ** 0.5}
        entry["judged@5"] = statistics.fmean((q, d) in llm_grades for q in query_ids for d in runs[(system, q)][:POOL_DEPTH])
        report["llm_judge"][system] = entry
    # Holm correction across every comparison in the table.
    compared = [s for s in systems if "vs_previous" in report["llm_judge"][s]]
    for system, adjusted in zip(compared, holm([report["llm_judge"][s]["vs_previous"]["p_value"] for s in compared])):
        report["llm_judge"][system]["vs_previous"]["p_holm"] = adjusted
    for system in systems:
        entry = report["llm_judge"][system]
        hub = entry["most_repeated_listing"]
        vs = entry.get("vs_previous")
        rows.append([f"`{system}`"] + [f"{entry[n]['mean']:.3f} [{entry[n]['ci95'][0]:.2f}, {entry[n]['ci95'][1]:.2f}]" for n in METRICS]
                    + [f"{entry['judged@5']:.0%}", f"{entry['non_food@5']:.1%}", f"{hub['queries']}× {hub['name'][:28]}",
                       f"{vs['delta_nDCG@5']:+.3f} vs `{vs['system']}` (p={vs['p_value']:.3f}, Holm {vs['p_holm']:.3f}, MDE {vs['mde_80']:.3f})" if vs else "—"])
    md += ["## LLM-judged results (all queries, mean [95% bootstrap CI])", "",
           "The nDCG@5 ideal comes from every judged listing for the query, across all systems (TREC practice), so adding a system can lower absolute scores but every system is divided by the same ideal. "
           "Judged@5 is the share of a system's top 5 that has a grade; ungraded listings count as 0. "
           "Δ compares with the named system; Holm adjusts p across all comparisons; MDE is the smallest difference "
           "detectable at 80% power.", "",
           table(rows, ["System", *METRICS, "Judged@5", "Non-food in top 5", "Most repeated listing", "Δ nDCG@5"]), ""]

    # 2. Ceiling: how many queries have any acceptable or exact listing in the whole pool.
    pool_best: dict[int, int] = defaultdict(int)
    for (query_id, _), grade in llm_grades.items():
        pool_best[query_id] = max(pool_best[query_id], grade)
    answerable = {"exact (3)": sum(pool_best[q] == 3 for q in query_ids), "acceptable (>=2)": sum(pool_best[q] >= 2 for q in query_ids)}
    report["answerability"] = answerable
    md += ["## Catalog ceiling", "", f"Queries whose judged pool contains an exact match: {answerable['exact (3)']}/{len(query_ids)}; "
           f"an acceptable match: {answerable['acceptable (>=2)']}/{len(query_ids)}. The remaining queries cannot score well with any system.", ""]

    # 3. Stratified by query type.
    if info:
        types = sorted({record["query_type"] for record in info.values()})
        rows, report["by_query_type"] = [], {}
        for system in systems:
            row = [f"`{system}`"]
            for query_type in types:
                values = [v for q, v in zip(query_ids, scores[system]["nDCG@5"]) if info[q]["query_type"] == query_type]
                report["by_query_type"].setdefault(system, {})[query_type] = statistics.fmean(values)
                row.append(f"{statistics.fmean(values):.3f}")
            rows.append(row)
        counts = Counter(info[q]["query_type"] for q in query_ids)
        md += ["## nDCG@5 by query type", "", table(rows, ["System", *[f"{t} (n={counts[t]})" for t in types]]), ""]

    # 3b. Vocabulary mismatch: queries containing words that never occur in the catalog
    # (often literal translations of English phrases) versus fully covered queries.
    vocabulary = {token for text in corpus.texts for token in re.findall(r"[a-z0-9]+", normalize_text(text))}
    unseen = {q: [t for t in re.findall(r"[a-z0-9]+", normalize_text(corpus.queries[q - 1]["text"])) if len(t) > 3 and t not in vocabulary]
              for q in query_ids}
    groups = {"has unseen word": [q for q in query_ids if unseen[q]], "all words in catalog": [q for q in query_ids if not unseen[q]]}
    report["vocabulary_mismatch"] = {"unseen_words": {q: words for q, words in unseen.items() if words},
                                     "by_group": {s: {g: statistics.fmean(v for q, v in zip(query_ids, scores[s]["nDCG@5"]) if q in qs)
                                                      for g, qs in groups.items() if qs} for s in systems}}
    md += ["## nDCG@5 by vocabulary overlap with the catalog", "",
           "A query has an unseen word when a word longer than three letters never occurs in any listing text "
           "(accent- and case-insensitive, no stemming).", "",
           table([[f"`{s}`", *[f"{report['vocabulary_mismatch']['by_group'][s].get(g, float('nan')):.3f}" for g in groups]] for s in systems],
                 ["System", *[f"{g} (n={len(qs)})" for g, qs in groups.items()]]), ""]

    # 4. Reranker confidence as an abstention signal.
    reranked = [record for record in run_records if record["system"] == "qe-hybrid-rerank"]
    if reranked:
        hit = dict(zip(query_ids, scores["qe-hybrid-rerank"]["Hit@5"]))
        # The cross-encoder outputs a logit; below zero means it rates every candidate as more likely irrelevant.
        low = [record["query_id"] for record in reranked if (record.get("max_rerank_score") or 0) < 0]
        high = [record["query_id"] for record in reranked if (record.get("max_rerank_score") or 0) >= 0]
        report["confidence"] = {"low_confidence_queries": len(low), "hit_low": statistics.fmean(hit[q] for q in low) if low else None,
                                "hit_high": statistics.fmean(hit[q] for q in high) if high else None}
        low_hit = f"{report['confidence']['hit_low']:.2f}" if low else "n/a"
        high_hit = f"{report['confidence']['hit_high']:.2f}" if high else "n/a"
        md += ["## Reranker confidence as a no-good-match signal", "",
               f"Queries where the reranker's best logit is below 0: {len(low)}. Judged Hit@5 for these: {low_hit}; "
               f"for the others: {high_hit}.", ""]

    # 5. Human validation of the judge.
    human = {}
    for record in read_jsonl(HUMAN):
        human[(record["query_id"], record["document_id"])] = record["grade"]
    if human:
        labelled = [(key, grade) for key, grade in human.items() if grade is not None and key in llm_grades]
        h = [grade for _, grade in labelled]
        j = [llm_grades[key] for key, _ in labelled]
        binary_h, binary_j = [int(g >= 2) for g in h], [int(g >= 2) for g in j]
        agreement = {
            "pairs": len(labelled), "unsure": sum(grade is None for grade in human.values()),
            "exact_agreement": sum(a == b for a, b in zip(h, j)) / len(h),
            "within_one": sum(abs(a - b) <= 1 for a, b in zip(h, j)) / len(h),
            "quadratic_weighted_kappa": cohen_kappa(h, j, [0, 1, 2, 3], "quadratic"),
            "binary_kappa_acceptable": cohen_kappa(binary_h, binary_j, [0, 1]),
            "confusion_human_rows_llm_cols": [[sum(1 for a, b in zip(h, j) if a == r and b == c) for c in range(4)] for r in range(4)],
        }
        # System ordering on fully labelled queries: human grades vs LLM grades on the same subset.
        # Systems added after the labelling round join once their pooled top 5 are labelled too.
        def fully_labelled(system_list: list[str]) -> list[int]:
            return [q for q in query_ids if all((q, d) in human for s in system_list for d in runs[(s, q)][:POOL_DEPTH])]

        # The labelling sample was drawn for the seven ladder systems; later systems join the comparison only
        # when their top 5 is fully labelled on those same queries, so the query set stays fixed.
        complete = fully_labelled([s for s in LADDER if s in systems])
        validated = [s for s in systems if complete and all((q, d) in human for q in complete for d in runs[(s, q)][:POOL_DEPTH])]
        agreement["validated_systems"] = validated
        if complete:
            human_grades = {key: (grade if grade is not None else 0) for key, grade in human.items()}
            human_scores = per_query_scores(runs, human_grades, validated, complete)
            subset_llm = per_query_scores(runs, llm_grades, validated, complete)
            agreement["complete_queries"] = len(complete)
            agreement["system_nDCG_human"] = {s: statistics.fmean(human_scores[s]["nDCG@5"]) for s in validated}
            agreement["system_nDCG_llm_same_queries"] = {s: statistics.fmean(subset_llm[s]["nDCG@5"]) for s in validated}
            agreement["kendall_tau_system_order"] = kendall_tau(list(agreement["system_nDCG_human"].values()),
                                                                list(agreement["system_nDCG_llm_same_queries"].values()))
        # Intra-annotator consistency: the same person relabels a sample blind; this bounds the
        # agreement any judge can be expected to reach with that person.
        retest = {(r["query_id"], r["document_id"]): r["grade"] for r in read_jsonl(HUMAN_RETEST)}
        matched = [key for key, grade in retest.items() if grade is not None and human.get(key) is not None]
        both = [(human[key], retest[key]) for key in matched]
        if both:
            first, second = [a for a, _ in both], [b for _, b in both]
            judged = [llm_grades.get(key) for key in matched]
            # The judge on the very same pairs: kappa depends on the grade mix, so only this is comparable.
            if all(grade is not None for grade in judged):
                agreement["judge_on_retest_pairs"] = {
                    "exact_agreement": sum(a == b for a, b in zip(first, judged)) / len(both),
                    "within_one": sum(abs(a - b) <= 1 for a, b in zip(first, judged)) / len(both),
                    "quadratic_weighted_kappa": cohen_kappa(first, judged, [0, 1, 2, 3], "quadratic"),
                    "binary_kappa_acceptable": cohen_kappa([int(a >= 2) for a in first], [int(b >= 2) for b in judged], [0, 1])}
            agreement["intra_annotator"] = {
                "pairs": len(both), "exact_agreement": sum(a == b for a, b in both) / len(both),
                "within_one": sum(abs(a - b) <= 1 for a, b in both) / len(both),
                "quadratic_weighted_kappa": cohen_kappa(first, second, [0, 1, 2, 3], "quadratic"),
                "binary_kappa_acceptable": cohen_kappa([int(a >= 2) for a in first], [int(b >= 2) for b in second], [0, 1])}
        # Unified human gold (pointwise + listing pairs + listwise rankings, Bradley-Terry per query):
        # how well do the judge's grades and the annotator's own grades order listings like the gold?
        gold = unified_gold()
        tau_judge, tau_own = [], []
        for q, strengths in gold.items():
            docs = [d for d in strengths if (q, d) in llm_grades]
            if len({llm_grades[(q, d)] for d in docs}) > 1:
                tau_judge.append(kendall_tau_b([strengths[d] for d in docs], [llm_grades[(q, d)] for d in docs]))
            own = [d for d in strengths if human.get((q, d)) is not None]
            if len({human[(q, d)] for d in own}) > 1:
                tau_own.append(kendall_tau_b([strengths[d] for d in own], [human[(q, d)] for d in own]))
        divider = divider_labels()
        in_context = [(label, human.get(key), llm_grades.get(key)) for key, label in divider.items()
                      if human.get(key) is not None and key in llm_grades]
        if tau_judge:
            agreement["unified_gold"] = {
                "queries": len(gold), "judge_tau_b": statistics.fmean(tau_judge), "own_pointwise_tau_b": statistics.fmean(tau_own),
                "divider_pairs": len(in_context),
                "divider_vs_own_grade_kappa": cohen_kappa([d for d, _, _ in in_context], [int(h >= 2) for _, h, _ in in_context], [0, 1]) if in_context else None,
                "divider_vs_judge_kappa": cohen_kappa([d for d, _, _ in in_context], [int(j >= 2) for _, _, j in in_context], [0, 1]) if in_context else None}
        report["human_validation"] = agreement
        md += ["## Judge validation against human labels", "",
               f"{agreement['pairs']} labelled pairs ({agreement['unsure']} marked unsure and excluded). "
               f"Exact agreement {agreement['exact_agreement']:.1%}, within one grade {agreement['within_one']:.1%}, "
               f"quadratic-weighted κ = {agreement['quadratic_weighted_kappa']:.2f}, "
               f"κ on acceptable (≥2) vs not = {agreement['binary_kappa_acceptable']:.2f}.", "",
               "Confusion matrix (rows: human grade 0–3, columns: LLM grade 0–3):", "",
               table([[r, *agreement["confusion_human_rows_llm_cols"][r]] for r in range(4)], ["human \\ LLM", "0", "1", "2", "3"]), ""]
        if "intra_annotator" in agreement:
            intra = agreement["intra_annotator"]
            md += ["Same annotator, blind second pass (the agreement ceiling), compared with the judge on the same pairs:", "",
                   table([["Annotator vs own retest", intra["pairs"], f"{intra['exact_agreement']:.1%}", f"{intra['within_one']:.1%}",
                           f"{intra['quadratic_weighted_kappa']:.2f}", f"{intra['binary_kappa_acceptable']:.2f}"],
                          *([["Annotator vs judge, same pairs", intra["pairs"], f"{j['exact_agreement']:.1%}", f"{j['within_one']:.1%}",
                              f"{j['quadratic_weighted_kappa']:.2f}", f"{j['binary_kappa_acceptable']:.2f}"]]
                            if (j := agreement.get("judge_on_retest_pairs")) else [])],
                         ["Comparison", "Pairs", "Exact", "Within one", "Weighted κ", "κ acceptable"]), ""]
        if "unified_gold" in agreement:
            u = agreement["unified_gold"]
            md += ["Unified human gold (pointwise grades, listing pairs, and listwise rankings combined per query with a "
                   f"Bradley-Terry model, {u['queries']} queries). Mean Kendall τ-b with the gold ordering: "
                   f"judge grades {u['judge_tau_b']:.2f}, the annotator's own pointwise grades {u['own_pointwise_tau_b']:.2f}. "
                   + (f"In-context acceptability (above the listwise divider) against grade ≥ 2, {u['divider_pairs']} listings: "
                      f"annotator κ = {u['divider_vs_own_grade_kappa']:.2f}, judge κ = {u['divider_vs_judge_kappa']:.2f}."
                      if u["divider_pairs"] else ""), ""]
        if complete:
            md += [f"System nDCG@5 on {len(complete)} fully labelled queries (unsure counted as 0):", "",
                   table([[f"`{s}`", f"{agreement['system_nDCG_human'][s]:.3f}", f"{agreement['system_nDCG_llm_same_queries'][s]:.3f}"] for s in validated],
                         ["System", "Human grades", "LLM grades"]), "",
                   f"Kendall τ between the two system orderings: {agreement['kendall_tau_system_order']:.2f}.", ""]

    # 5b. Preference judgments: whole-list side-by-side by the judge, and single listing pairs by human and judge.
    llm_sxs = read_jsonl(OUT / "llm_pairwise.jsonl")
    if llm_sxs:
        with redirect_stdout(io.StringIO()):
            sxs = pairwise_summary(llm_sxs, "llm")
            items = pairwise_summary_items() if (OUT / "human_item_pairwise.jsonl").is_file() else {}
        consistent = sum(record["votes"][0] == record["votes"][1] for record in llm_sxs)
        report["pairwise"] = {"llm_list_sxs": sxs, "order_consistent": consistent, "comparisons": len(llm_sxs), "item_pairs": items}
        md += ["## Preference judgments", "",
               f"Side by side: the judge sees two top-5 lists for a query in both orders; a preference counts only when "
               f"both orders agree ({consistent}/{len(llm_sxs)} comparisons were order-consistent, including ties).", "",
               table([[name, v["wins"], v["ties"], v["losses"], f"{v['p_sign_test']:.3f}"] for name, v in sxs.items()],
                     ["Comparison", "Wins", "Ties", "Losses", "Sign test p"]), ""]
        if items:
            md += ["Single listing pairs, picked by how the two listings were graded: *tied* (same human grade), "
                   "*conflict* (human and judge grades order them differently), *clear* (grades differ by 2+). "
                   "The annotator picked the better listing; the judge did the same in both orders.", "",
                   table([[kind, v["pairs"], v["human_preference_decided"],
                           # Tied pairs have equal own grades by construction, so the grades cannot pick a side.
                           "n/a" if kind == "tied" else f"{v['agrees_with_own_grades']}/{v['human_preference_decided']}",
                           f"{v['agrees_with_judge_grades']}/{v['human_preference_decided']}",
                           f"{v['llm_preference_matches_human']}/{v['llm_compared']}"] for kind, v in items.items()],
                         ["Kind", "Pairs", "Human decided", "Agrees with own grades", "Agrees with judge grades",
                          "Judge preference = human"]), ""]

    # 5c. Does the listing photo help the judge? Same model and prompt, photo attached.
    multimodal = OUT / "judge_mm.json"
    if multimodal.is_file():
        mm = json.loads(multimodal.read_text())
        report["multimodal_judge"] = mm
        rows = []
        for subset, label in (("all", "all labelled pairs"), ("with_photo", "pairs with a photo")):
            for judge_name in ("text_judge", "multimodal_judge"):
                v = mm[subset][judge_name]
                rows.append([label, judge_name.replace("_", " "), v["pairs"], f"{v['exact']:.1%}", f"{v['weighted_kappa']:.2f}",
                             f"{v['kappa_acceptable']:.2f}"])
        md += ["## Multimodal judge (listing photo attached)", "",
               "The same judge model reads the listing photo as well as the text; the prompt differs in one line. "
               "Agreement with the human grades:", "",
               table(rows, ["Pairs", "Judge", "n", "Exact", "Weighted κ", "κ acceptable"]), "",
               f"With the photo the judge moved closer to the human grade on {mm['with_photo']['closer']} pairs and further on "
               f"{mm['with_photo']['further']}. Kendall τ-b with the unified human gold: text "
               f"{mm['unified_gold_tau_b'].get('text_judge', float('nan')):.2f}, multimodal "
               f"{mm['unified_gold_tau_b'].get('multimodal_judge', float('nan')):.2f}"
               + (f". κ with the listwise divider ({mm['divider_kappa']['pairs']} listings): text {mm['divider_kappa']['text_judge']:.2f}, "
                  f"multimodal {mm['divider_kappa']['multimodal_judge']:.2f}." if "divider_kappa" in mm else "."), ""]

    # 5d. Behavioural ground truth: held-out historical search terms, no LLM involved.
    behaviour = OUT / "behavioral.json"
    if behaviour.is_file():
        bh = json.loads(behaviour.read_text())
        report["behavioral"] = {k: v for k, v in bh.items() if k != "systems"}
        report["behavioral"]["systems"] = {s: {k: v for k, v in r.items() if k != "per_term_mrr"} for s, r in bh["systems"].items()}
        listed = [s for s in systems if s in bh["systems"]]
        md += ["## Behavioural check (held-out search terms)", "",
               f"{bh.get('held_out_terms')} historical search terms (a hashed 20% split, never used by any system) with the "
               "listings users reached after searching them. Labels are incomplete, so these are recall-style numbers; "
               "systems that call the LLM run on a fixed random sample of "
               f"{bh.get('llm_sample')} terms.", "",
               table([[f"`{s}`", bh["systems"][s]["terms"], f"{bh['systems'][s]['hit@10']:.3f}",
                       f"{bh['systems'][s]['mrr@10']:.3f} [{bh['systems'][s]['mrr@10_ci95'][0]:.3f}, {bh['systems'][s]['mrr@10_ci95'][1]:.3f}]"]
                      for s in listed], ["System", "Terms", "Hit@10", "MRR@10 [95% CI]"]), ""]
        if bh.get("kendall_tau_vs_judge") is not None:
            md += [f"Kendall τ between the behavioural ordering (same-sample MRR@10) and the judge's nDCG@5 ordering: "
                   f"{bh['kendall_tau_vs_judge']:.2f}.", ""]
        ablation = bh.get("behavior_ablation") or {}
        paired_rows = {**{s: v for s, v in ablation.items() if isinstance(v, dict)}, **bh.get("vl_comparisons", {}),
                       **bh.get("two_tower_comparisons", {})}
        if paired_rows:
            md += ["Paired MRR@10 on the same held-out terms. The `-logs` systems append past search terms from the training "
                   f"split to the listing text ({ablation.get('listings_expanded', 0)} listings expanded; "
                   f"{ablation.get('eval_queries_in_training_terms', 0)} evaluation queries appear among those terms). The VL "
                   "systems matter here because the judge shares their model family.", "",
                   table([[f"`{s}`", f"`{v['baseline']}`", f"{v['mrr@10']:.3f}", f"{v['baseline_mrr@10']:.3f}", f"{v['delta']:+.3f}",
                           f"{v['p_value']:.4f}"] for s, v in paired_rows.items()],
                         ["System", "Compared with", "MRR@10", "Baseline MRR@10", "Δ", "Paired p"]), ""]

    # 6. Models, compute, and cost.
    llm = LLM(offline=True)
    hf_models = {}
    for name in ("multilingual-e5-small", "harrier-oss-v1-0.6b", "llama-nemotron-rerank-1b-v2"):
        manifest = ROOT / ".demo-cache/models" / name / "prosus-model.json"
        if manifest.is_file():
            hf_models[name] = json.loads(manifest.read_text())
    judge_model = judgments[0]["judge"] if judgments else None
    report["models"] = {"llm": [llm.describe(m) for m in dict.fromkeys(filter(None, [UNDERSTAND_MODEL, ANNOTATION_MODEL, judge_model]))],
                        "huggingface": hf_models}
    md += ["## Models", "", table([[m["model"], m.get("quantization", ""), (m.get("digest") or "")[:12]] for m in report["models"]["llm"]],
                                  ["LLM", "Quantization", "Digest"]), "",
           table([[v["repo_id"], v["revision"][:12]] for v in hf_models.values()], ["Hugging Face model", "Revision"]), ""]
    usage = usage_summary()
    report["usage"] = usage
    latency = {s: statistics.median(r["elapsed_ms"] for r in run_records if r["system"] == s) for s in systems}
    report["median_elapsed_ms"] = latency
    md += ["## LLM usage (cache misses only)", "",
           table([[task, int(v["calls"]), int(v["input_tokens"]), int(v["output_tokens"]), f"{v['seconds'] / 60:.1f} min",
                   f"${v['cost_usd']:.3f}"] for task, v in sorted(usage["by_task"].items())],
                 ["Task · model", "Calls", "Input tokens", "Output tokens", "Wall time (summed over parallel calls)", "API cost"]), "",
           f"Total API cost: ${usage['total_cost_usd']:.2f} (local models cost nothing per call).", ""]
    benchmark = OUT / "latency.json"
    if benchmark.is_file():
        bench = json.loads(benchmark.read_text())
        report["latency"] = bench
        md += ["## Latency (controlled, one request at a time)", "", bench["note"], "",
               table([[f"`{s}`", f"{v['median']:.0f}", f"{v['p90']:.0f}"] for s, v in bench["systems_ms"].items()],
                     ["System", "Median ms", "p90 ms"]), "",
               f"Uncached LLM query understanding: median {bench['understanding_uncached_s']['median']:.1f} s "
               f"over {bench['understanding_uncached_s']['samples']} queries.", ""]

    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "report.json").write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    (OUT / "report.md").write_text("\n".join(md), encoding="utf-8")
    print("\n".join(md))


if __name__ == "__main__":
    main()
