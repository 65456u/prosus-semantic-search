"""Offline tests for generative-retrieval identifiers and constrained decoding (no model)."""

from __future__ import annotations

from retrieval.generative.decode import IdTrie
from retrieval.generative.ids import unique_ids


class WordTokenizer:
    """One token per word, enough to test length capping."""

    def __call__(self, text, add_special_tokens=False):
        return {"input_ids": list(range(len(text.split())))}


def test_trie_allows_only_existing_identifiers_and_maps_back():
    trie = IdTrie([[5, 6, 7], [5, 8], [9]], eos_token_id=1)
    assert sorted(trie.allowed([])) == [5, 9]
    assert sorted(trie.allowed([5])) == [6, 8]
    assert trie.allowed([5, 8]) == [1]  # complete identifier: only EOS
    assert trie.allowed([4]) == [1]  # an impossible prefix can only finish
    assert trie.lookup[(5, 8)] == 1 and trie.lookup[(9,)] == 2


def test_unique_ids_cap_length_and_suffix_repeats():
    long = " ".join(["palavra"] * 60)
    ids = unique_ids(["pizza | calabresa", "pizza | calabresa", long, "açaí"], WordTokenizer())
    assert ids[:2] == ["pizza | calabresa", "pizza | calabresa #2"]
    assert len(ids[2].split()) == 40
    assert len(set(ids)) == len(ids)
