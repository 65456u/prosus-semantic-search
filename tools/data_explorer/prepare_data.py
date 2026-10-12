"""Build a local browser dataset from the supplied CSVs. Source files stay intact."""
import csv
import json
from collections import Counter
from pathlib import Path

HERE = Path(__file__).resolve().parent
SOURCE = HERE.parents[1] / "data"


def read_csv(name):
    with (SOURCE / name).open(encoding="utf-8-sig", newline="") as stream:
        return list(csv.DictReader(stream))


def main():
    rows = read_csv("5k_items_curated.csv")
    items = []
    for number, row in enumerate(rows, 1):
        items.append({
            "row": number,
            "_id": row["_id"],
            "itemId": row["itemId"],
            "merchantId": row["merchantId"],
            "itemMetadata": json.loads(row["itemMetadata"]),
            "itemProfile": json.loads(row["itemProfile"]),
        })
    queries = [row["search_term_pt"] for row in read_csv("queries.csv")]
    ids = Counter(item["itemId"] for item in items)
    categories = Counter(item["itemMetadata"]["taxonomy"].get("l0", "OUTROS") for item in items)
    stats = {
        "rows": len(items), "uniqueItems": len(ids),
        "merchants": len({item["merchantId"] for item in items}),
        "queries": len(queries),
        "descriptions": sum(bool(item["itemMetadata"]["description"].strip()) for item in items),
        "images": sum(bool(item["itemMetadata"]["images"]) for item in items),
        "search": sum(bool(item["itemProfile"]["search"]) for item in items),
        "coPurchase": sum(bool(item["itemProfile"]["metrics"]["coPurchaseItems"]) for item in items),
        "customizations": sum(bool(item["itemProfile"]["metrics"]["customizations"]) for item in items),
        "duplicateRows": sum(count - 1 for count in ids.values()),
        "duplicateIds": [item_id for item_id, count in ids.items() if count > 1],
        "categories": [{"key": key, "count": count} for key, count in categories.most_common()],
        "flags": {key: sum(item["itemMetadata"][key] is True for item in items) for key in ["vegan", "lacFree", "organic"]},
    }
    data = {"items": items, "queries": queries, "stats": stats}
    text = json.dumps(data, ensure_ascii=False, separators=(",", ":"), allow_nan=False)
    (HERE / "data.js").write_text("window.PROSUS_DATA=" + text + ";\n", encoding="utf-8")
    print(f"Loaded {len(items):,} rows, {len(ids):,} item IDs, and {len(queries)} queries.")


if __name__ == "__main__":
    main()
