"""Export all supplied queries through the same API used by the frontend.

This is functional validation and result export, not a relevance evaluation or
controlled throughput benchmark. Each response records its query-cache state.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from urllib.request import Request, urlopen


def request(base: str, path: str, payload: dict | None = None):
    data = None if payload is None else json.dumps(payload).encode()
    req = Request(base.rstrip("/") + path, data=data, headers={"Content-Type": "application/json"})
    with urlopen(req, timeout=60) as response:
        return json.load(response)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--url", default="http://127.0.0.1:8000")
    parser.add_argument("--output", type=Path, default=Path("output/demo/results.jsonl"))
    parser.add_argument("--methods", nargs="+", choices=["bm25", "e5", "hybrid"], default=["bm25", "e5", "hybrid"])
    parser.add_argument("--no-query-cache", action="store_true")
    args = parser.parse_args()
    status = request(args.url, "/api/status")
    ready = {method["id"]: method["ready"] for method in status["methods"]}
    if any(not ready[method] for method in args.methods):
        raise SystemExit("Wait until the requested models are ready, then run again.")
    queries = request(args.url, "/api/queries")["queries"]
    args.output.parent.mkdir(parents=True, exist_ok=True)
    successful, failed, short_results = 0, [], []
    with args.output.open("w", encoding="utf-8") as output:
        for number, query in enumerate(queries, 1):
            response = request(args.url, "/api/compare", {
                "query": query["text"], "methods": args.methods, "top_k": 5,
                "use_query_cache": not args.no_query_cache,
            })
            for run in response["runs"]:
                record = {"query_id": query["id"], "query": query["text"], **run}
                if "error" in run:
                    failed.append({"query_id": query["id"], "method": run["method"], "error": run["error"]})
                else:
                    results = run["results"]
                    listing_ids = {(item["item_id"], item["merchant_id"]) for item in results}
                    if len(results) != 5 or len(listing_ids) != 5:
                        short_results.append({"query_id": query["id"], "method": run["method"]})
                    else:
                        successful += 1
                output.write(json.dumps(record, ensure_ascii=False) + "\n")
            if number % 20 == 0:
                print(f"Validated {number}/{len(queries)} queries", flush=True)
    summary = {
        "purpose": "functional validation; no ground-truth relevance labels",
        "queries": len(queries), "methods": args.methods,
        "successful_top5_runs": successful, "errors": failed,
        "invalid_top5_runs": short_results,
        "query_cache_enabled": not args.no_query_cache,
        "model_revision": status["semantic"].get("revision"),
        "device": status["semantic"].get("device"),
        "results_path": str(args.output),
    }
    args.output.with_suffix(".summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n")
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    if failed or short_results:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
