"""Trie-constrained beam search: the decoder can only spell identifiers that exist.

Training normalises the output softmax over the identifier vocabulary only (every token
that occurs in some identifier, plus EOS), because the full 262k-token softmax does not fit
in memory. Decoding applies the same normalisation, then masks tokens the trie does not
allow, so sequence scores mean the same thing in training and inference.
"""

from __future__ import annotations

from typing import Any


class IdTrie:
    """Token-level prefix tree over every identifier followed by EOS."""

    def __init__(self, sequences: list[list[int]], eos_token_id: int):
        self.root: dict[int, dict] = {}
        self.eos = eos_token_id
        self.lookup: dict[tuple[int, ...], int] = {}
        self.vocabulary = sorted({token for tokens in sequences for token in tokens} | {eos_token_id})
        for index, tokens in enumerate(sequences):
            node = self.root
            for token in [*tokens, eos_token_id]:
                node = node.setdefault(token, {})
            self.lookup[tuple(tokens)] = index

    def allowed(self, prefix: list[int]) -> list[int]:
        node = self.root
        for token in prefix:
            node = node.get(token)
            if node is None:
                return [self.eos]  # a finished beam only continues with EOS/padding
        return list(node) or [self.eos]


def encode_targets(tokenizer, ids: list[str]) -> list[list[int]]:
    return [tokenizer(text, add_special_tokens=False)["input_ids"] for text in ids]


def trie_processor(trie: IdTrie, special: set[int], device: str):
    """Renormalise log-probabilities over the identifier vocabulary, then keep only trie-allowed tokens."""
    import torch
    from transformers import LogitsProcessor

    subset = torch.tensor(trie.vocabulary, device=device)

    class TrieLogitsProcessor(LogitsProcessor):
        def __call__(self, input_ids: Any, scores: Any) -> Any:
            restricted = scores[:, subset]
            restricted = restricted - torch.logsumexp(restricted, dim=-1, keepdim=True)
            renormalised = torch.full_like(scores, float("-inf"))
            renormalised[:, subset] = restricted
            output = torch.full_like(scores, float("-inf"))
            for row, decoder_ids in enumerate(input_ids.tolist()):
                prefix = list(decoder_ids)
                while prefix and prefix[0] in special:
                    prefix = prefix[1:]
                allowed = trie.allowed(prefix)
                output[row, allowed] = renormalised[row, allowed]
            return output

    return TrieLogitsProcessor()


def generate(model, tokenizer, trie: IdTrie, texts: list[str], beams: int, top: int, max_input: int = 64,
             max_new_tokens: int = 48, device: str = "mps") -> list[list[tuple[int, float]]]:
    """For each input text, up to `top` (identifier index, sequence score) pairs, best first."""
    import torch
    from transformers import LogitsProcessorList

    special = {tokenizer.pad_token_id, model.generation_config.decoder_start_token_id}
    processors = LogitsProcessorList([trie_processor(trie, special, device)])
    batch = tokenizer(texts, padding=True, truncation=True, max_length=max_input, return_tensors="pt").to(device)
    with torch.inference_mode():
        output = model.generate(**batch, num_beams=beams, num_return_sequences=min(top, beams), max_new_tokens=max_new_tokens,
                                logits_processor=processors, output_scores=True, return_dict_in_generate=True,
                                early_stopping=True, length_penalty=0.0)
    per_input = min(top, beams)
    results: list[list[tuple[int, float]]] = []
    for i in range(len(texts)):
        ranked = []
        for j in range(per_input):
            sequence = [int(t) for t in output.sequences[i * per_input + j].tolist()]
            sequence = [t for t in sequence if t not in special]
            if trie.eos in sequence:
                sequence = sequence[:sequence.index(trie.eos)]
            index = trie.lookup.get(tuple(sequence))
            if index is not None and all(index != seen for seen, _ in ranked):
                ranked.append((index, float(output.sequences_scores[i * per_input + j])))
        results.append(ranked)
    return results
