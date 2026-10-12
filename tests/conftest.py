"""Offline fixtures for testing retrieval without downloading an ML model."""

from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path

import numpy as np
import pytest


ROOT = Path(__file__).resolve().parents[1]


def write_corpus(path: Path, descriptions: list[str] | None = None) -> tuple[Path, Path]:
    """Use the assignment's actual CSV schema for a small independent corpus."""
    names = ["Pão de queijo", "Pizza de tomate", "Sorvete de morango", "Arroz integral"]
    descriptions = descriptions or ["Queijo e massa assada", "Forno a lenha", "Gelado", "Pacote 500g"]
    path.mkdir(parents=True, exist_ok=True)
    items_path = path / "items.csv"
    with items_path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=["_id", "itemId", "itemMetadata", "itemProfile", "merchantId"])
        writer.writeheader()
        for index, (name, description) in enumerate(zip(names, descriptions, strict=True)):
            writer.writerow({
                "_id": f"document-{index}",
                "itemId": f"item-{index}",
                "merchantId": "merchant-1",
                "itemMetadata": json.dumps({"name": name, "description": description,
                                             "category_name": "Comida", "price": 10.0}, ensure_ascii=False),
                "itemProfile": json.dumps({"search": [], "metrics": {}}),
            })
    queries_path = path / "queries.csv"
    with queries_path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=["search_term_pt"])
        writer.writeheader()
        writer.writerow({"search_term_pt": "pão de queijo"})
    return items_path, queries_path


@pytest.fixture
def tiny_csvs(tmp_path):
    return write_corpus(tmp_path / "source")


class DummyEncoder:
    """A deterministic encoder which records actual inference, including prefixes."""

    def __init__(self, variant: str = "model-a"):
        self.variant = variant
        self.calls: list[list[str]] = []

    def encode(self, sentences, **kwargs):
        sentences = [sentences] if isinstance(sentences, str) else list(sentences)
        self.calls.append(sentences)
        vectors = []
        for sentence in sentences:
            digest = hashlib.sha256((self.variant + sentence).encode("utf-8")).digest()
            vector = np.frombuffer(digest[:16], dtype=np.uint8).astype(np.float32) + 1
            vectors.append(vector / np.linalg.norm(vector))
        return np.asarray(vectors, dtype=np.float32)

    @property
    def inputs(self) -> list[str]:
        return [sentence for batch in self.calls for sentence in batch]
