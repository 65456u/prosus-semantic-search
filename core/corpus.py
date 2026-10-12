"""Load the assignment data without dropping or merging source records.

Every row has a stable, one-based document ID. Retrieval can deduplicate listings,
but catalog browsing and item details retain all source rows.
"""

from __future__ import annotations

import csv
import json
import re
import unicodedata
from pathlib import Path
from typing import Any
from urllib.parse import quote


def normalize_text(text: str) -> str:
    """Case- and accent-insensitive text used only for lexical matching."""
    decomposed = unicodedata.normalize("NFKD", str(text).casefold())
    return "".join(char for char in decomposed if not unicodedata.combining(char))


def image_url(path: str | None) -> str | None:
    if not path:
        return None
    if path.startswith(("http://", "https://")):
        return path
    return "https://static.ifood-static.com.br/image/upload/t_low/pratos/" + quote(path, safe="/")


_DIETARY_LABELS = {
    "DIET": "diet",
    "GLUTEN_FREE": "sem glúten",
    "LAC_FREE": "sem lactose",
    "ORGANIC": "orgânico",
    "SUGAR_FREE": "sem açúcar",
    "VEGAN": "vegano",
    "VEGETARIAN": "vegetariano",
    "ZERO": "zero",
}
_DISH_LABELS = {"ALCOHOLIC_DRINK": "bebida alcoólica", "FROSTY": "gelado", "NATURAL": "natural"}


def build_item_text(metadata: dict[str, Any]) -> str:
    """One common Portuguese representation for BM25 and dense retrieval.

Only product metadata is used. Historical searches, purchases, behavioral
metrics, and prices are excluded. False dietary flags are not evidence that a
product contains an ingredient, so only explicitly positive flags are added.
Taxonomy labels are kept as supplied, with underscores converted to spaces.
"""
    parts = [str(metadata.get(key) or "").strip() for key in ("name", "description", "category_name")]
    taxonomy = metadata.get("taxonomy") or {}
    parts.extend(str(taxonomy.get(level) or "").replace("_", " ").strip() for level in ("l0", "l1", "l2"))
    attributes = []
    for key, label in (("vegan", "vegano"), ("lacFree", "sem lactose"), ("organic", "orgânico")):
        if metadata.get(key) is True:
            attributes.append(label)
    for tag in metadata.get("tags") or []:
        values = tag.get("value") or []
        if isinstance(values, str):
            values = [values]
        for value in values:
            value = str(value).strip()
            key = tag.get("key")
            if key == "DIETARY_RESTRICTIONS" and value in _DIETARY_LABELS:
                attributes.append(_DIETARY_LABELS[value])
            elif key == "DISH_CLASSIFICATION" and value in _DISH_LABELS:
                attributes.append(_DISH_LABELS[value])
            elif key == "PORTION_SIZE" and re.fullmatch(r"SERVES_[1-4]", value):
                attributes.append(f"serve {value[-1]} pessoa" + ("s" if value[-1] != "1" else ""))
            elif key == "PIZZA_SIZE" and value:
                attributes.append("pizza " + value)
    parts.extend(dict.fromkeys(attributes))
    return "\n".join(dict.fromkeys(part for part in parts if part))


class Corpus:
    def __init__(self, items_path: str | Path, queries_path: str | Path):
        self.items: list[dict[str, Any]] = []
        self.queries: list[dict[str, Any]] = []
        self.texts: list[str] = []
        self.document_ids: list[str] = []
        self._details: dict[str, dict[str, Any]] = {}
        self._search_texts: list[str] = []
        with Path(items_path).open(encoding="utf-8-sig", newline="") as source:
            for row_number, source_row in enumerate(csv.DictReader(source), start=1):
                metadata = json.loads(source_row.get("itemMetadata") or "{}")
                profile = json.loads(source_row.get("itemProfile") or "{}")
                document_id = f"row-{row_number:06d}"
                images = metadata.get("images") or []
                item = {
                    "document_id": document_id,
                    "row": row_number,
                    "item_id": source_row["itemId"],
                    "merchant_id": source_row["merchantId"],
                    "name": metadata.get("name") or "",
                    "description": metadata.get("description") or "",
                    "category": metadata.get("category_name") or "",
                    "taxonomy_l0": (metadata.get("taxonomy") or {}).get("l0") or "",
                    "price": metadata.get("price"),
                    "images": images,
                    "image_url": image_url(images[0]) if images else None,
                    "vegan": metadata.get("vegan") is True,
                    "lac_free": metadata.get("lacFree") is True,
                    "organic": metadata.get("organic") is True,
                }
                text = build_item_text(metadata)
                self.items.append(item)
                self.texts.append(text)
                self.document_ids.append(document_id)
                self._search_texts.append(normalize_text(text))
                self._details[document_id] = {
                    **item,
                    "source_id": source_row.get("_id"),
                    "item_metadata": metadata,
                    "item_profile": profile,
                    "retrieval_text": text,
                }
        with Path(queries_path).open(encoding="utf-8-sig", newline="") as source:
            self.queries = [
                {"id": index, "text": row["search_term_pt"]}
                for index, row in enumerate(csv.DictReader(source), start=1)
            ]
        self.categories = sorted({item["taxonomy_l0"] for item in self.items if item["taxonomy_l0"]})
        self.counts = {
            "items": len(self.items),
            "unique_items": len({item["item_id"] for item in self.items}),
            "queries": len(self.queries),
            "merchants": len({item["merchant_id"] for item in self.items}),
        }

    def get_item(self, document_id: str) -> dict[str, Any]:
        """Return product metadata and full original profile; raise KeyError if absent."""
        return self._details[document_id]

    def catalog(self, query: str = "", category: str = "", offset: int = 0, limit: int = 24) -> dict[str, Any]:
        """Browse source rows with simple text filtering, not relevance ranking."""
        if offset < 0 or limit < 1:
            raise ValueError("Catalog offset must be non-negative and limit must be positive.")
        terms = normalize_text(query).split()
        filtered = [
            item for item, text in zip(self.items, self._search_texts)
            if (not category or category in (item["taxonomy_l0"], item["category"]))
            and all(term in text for term in terms)
        ]
        return {
            "items": filtered[offset:offset + limit],
            "total": len(filtered),
            "offset": offset,
            "limit": limit,
            "categories": self.categories,
        }
