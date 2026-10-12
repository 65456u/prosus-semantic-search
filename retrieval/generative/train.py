"""Fine-tune T5Gemma 2 (270M-270M) to decode item identifiers; select the checkpoint on the dev set.

    .venv/bin/python -m retrieval.generative.train --variant semantic [--epochs 4]

Fine-tunes everything except the shared token embeddings (identifiers are spelled with
existing tokens, so the 262k x 640 embedding table stays frozen), with bf16 autocast on
Apple MPS. Do not run it while a local LLM is loaded: together they exceed 16 GB and swap.
After every epoch, trie-constrained
beam search on a fixed development sample reports Recall@1/@10; the best epoch is saved
to .demo-cache/genret/<variant>/ and its metrics to output/genret/train-<variant>.json.
"""

from __future__ import annotations

import argparse
import gc
import json
import math
import random
import time

from core.local_models import pick_device
from core.common import ROOT, load_corpus

from .data import build_examples
from .decode import IdTrie, encode_targets, generate
from .ids import IDS, TOKENIZER_DIR, VARIANTS

CHECKPOINTS = ROOT / ".demo-cache" / "genret"


def load_model(device: str, checkpoint=None):
    """Text-only T5Gemma 2 in fp32. The 417M-parameter vision tower is never used, so it is
    replaced by an identity module; checkpoints store only the remaining (text) weights."""
    import torch
    from safetensors.torch import load_file
    from transformers import AutoTokenizer, T5Gemma2ForConditionalGeneration

    tokenizer = AutoTokenizer.from_pretrained(str(TOKENIZER_DIR))
    model = T5Gemma2ForConditionalGeneration.from_pretrained(str(TOKENIZER_DIR))
    model.model.encoder.vision_tower = torch.nn.Identity()
    model = model.float()  # some sub-modules load as bf16 regardless of the requested dtype
    model.generation_config.decoder_start_token_id = model.config.decoder.bos_token_id
    if checkpoint is not None:
        model.load_state_dict(load_file(str(checkpoint / "model.safetensors")), strict=False)
    return tokenizer, model.to(device)


def save_checkpoint(model, path) -> None:
    from safetensors.torch import save_file

    path.mkdir(parents=True, exist_ok=True)
    state = {name: tensor.detach().cpu().contiguous() for name, tensor in model.state_dict().items() if "vision_tower" not in name}
    save_file(state, str(path / "model.safetensors"))


def release(device: str) -> None:
    """Return cached accelerator memory after evaluation or saving. On a 16 GB Mac the
    cache left by beam search pushed training into swap (2.4 s/step became 85 s/step)."""
    import torch

    gc.collect()
    if device == "mps":
        torch.mps.empty_cache()


def recall(model, tokenizer, trie: IdTrie, pairs: list[tuple[str, int]], device: str, beams: int = 10) -> dict[str, float]:
    model.eval()
    hits1 = hits10 = 0
    for start in range(0, len(pairs), 16):
        chunk = pairs[start:start + 16]
        ranked = generate(model, tokenizer, trie, [text for text, _ in chunk], beams=beams, top=10, device=device)
        for (_, target), results in zip(chunk, ranked):
            indices = [index for index, _ in results]
            hits1 += bool(indices) and indices[0] == target
            hits10 += target in indices
    model.train()
    return {"recall@1": hits1 / len(pairs), "recall@10": hits10 / len(pairs)}


def main() -> None:
    import torch

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--variant", choices=VARIANTS, required=True)
    parser.add_argument("--epochs", type=int, default=4)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--max-input", type=int, default=64)
    parser.add_argument("--dev-sample", type=int, default=300)
    parser.add_argument("--max-steps", type=int, default=0, help="stop early (smoke tests)")
    parser.add_argument("--fp32", action="store_true", help="disable bf16 autocast")
    parser.add_argument("--tag", default="", help="suffix for the checkpoint and system name, e.g. -long")
    parser.add_argument("--save-every", type=int, default=0, help="also keep a checkpoint every N epochs (<variant><tag>-e<epoch>)")
    parser.add_argument("--patience", type=int, default=0, help="stop after N epochs without a better dev Recall@10")
    parser.add_argument("--stop-after-epoch", type=int, default=0,
                        help="stop after this epoch, keeping the --epochs learning-rate schedule (equal-budget comparisons)")
    args = parser.parse_args()
    torch.manual_seed(0)
    device = pick_device()
    corpus = load_corpus()
    ids = json.loads(IDS.read_text(encoding="utf-8"))
    identifiers = ids["variants"][args.variant]
    train, dev, counts = build_examples(corpus, ids, args.variant)
    dev_sample = random.Random(1).sample(dev, min(args.dev_sample, len(dev)))
    tokenizer, model = load_model(device)
    targets = encode_targets(tokenizer, identifiers)
    trie = IdTrie(targets, tokenizer.eos_token_id)
    model.get_input_embeddings().weight.requires_grad_(False)  # tied with the output projection
    # Softmax over the identifier vocabulary only; the full 262k-token softmax runs out of memory.
    subset = torch.tensor(trie.vocabulary, device=device)
    position = torch.full((model.config.decoder.vocab_size,), -100, dtype=torch.long)
    position[torch.tensor(trie.vocabulary)] = torch.arange(len(trie.vocabulary))
    output_rows = model.lm_head.out_proj.weight.detach()[subset]
    trainable = [parameter for parameter in model.parameters() if parameter.requires_grad]
    optimizer = torch.optim.AdamW(trainable, lr=args.lr, weight_decay=0.01)
    steps_per_epoch = math.ceil(len(train) / args.batch_size)
    total = steps_per_epoch * args.epochs if not args.max_steps else args.max_steps
    warmup = min(200, total // 10)

    def schedule(step: int) -> float:  # linear warm-up, then linear decay to zero
        return (step + 1) / max(1, warmup) if step < warmup else max(0.0, (total - step) / max(1, total - warmup))

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, schedule)
    print(f"{args.variant}: {counts}; {steps_per_epoch} steps/epoch on {device}; "
          f"{sum(p.numel() for p in trainable) / 1e6:.0f}M trainable parameters", flush=True)
    history, best, best_epoch, step = [], -1.0, 0, 0
    output = ROOT / "output" / "genret" / f"train-{args.variant}{args.tag}.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    model.train()
    for epoch in range(1, args.epochs + 1):
        random.Random(epoch).shuffle(train)
        started, running = time.perf_counter(), 0.0
        for start in range(0, len(train), args.batch_size):
            chunk = train[start:start + args.batch_size]
            inputs = tokenizer([text for text, _ in chunk], padding=True, truncation=True, max_length=args.max_input, return_tensors="pt")
            labels = [targets[target] + [tokenizer.eos_token_id] for _, target in chunk]
            width = max(map(len, labels))
            label_tensor = torch.tensor([sequence + [-100] * (width - len(sequence)) for sequence in labels])
            decoder_input_ids = model.prepare_decoder_input_ids_from_labels(label_tensor).to(device)
            targets_in_subset = torch.where(label_tensor >= 0, position[label_tensor.clamp(min=0)], -100).to(device)
            with torch.autocast(device_type=device, dtype=torch.bfloat16, enabled=not args.fp32 and device != "cpu"):
                hidden = model.model(input_ids=inputs["input_ids"].to(device), attention_mask=inputs["attention_mask"].to(device),
                                     decoder_input_ids=decoder_input_ids).last_hidden_state
                logits = hidden @ output_rows.T
            loss = torch.nn.functional.cross_entropy(logits.float().reshape(-1, logits.shape[-1]), targets_in_subset.reshape(-1),
                                                     ignore_index=-100)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(trainable, 1.0)
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad(set_to_none=True)
            running = 0.98 * running + 0.02 * float(loss) if step else float(loss)
            step += 1
            if step % 100 == 0:
                print(f"epoch {epoch} step {step}/{total} loss {running:.3f} ({(time.perf_counter() - started) / (start // args.batch_size + 1):.2f}s/step)", flush=True)
            if args.max_steps and step >= args.max_steps:
                break
        metrics = {"epoch": epoch, "loss": running, "minutes": (time.perf_counter() - started) / 60,
                   **recall(model, tokenizer, trie, dev_sample, device)}
        release(device)
        history.append(metrics)
        print(json.dumps(metrics), flush=True)
        if metrics["recall@10"] > best:
            best, best_epoch = metrics["recall@10"], epoch
            save_checkpoint(model, CHECKPOINTS / f"{args.variant}{args.tag}")
            release(device)
        if args.save_every and epoch % args.save_every == 0:
            save_checkpoint(model, CHECKPOINTS / f"{args.variant}{args.tag}-e{epoch}")
            release(device)
        output.write_text(json.dumps({"variant": args.variant, "counts": counts, "args": vars(args), "history": history,
                                      "best_recall@10": best, "best_epoch": best_epoch}, indent=2), encoding="utf-8")
        if args.max_steps and step >= args.max_steps or epoch == args.stop_after_epoch:
            break
        if args.patience and epoch - best_epoch >= args.patience:
            print(f"early stop: no better dev Recall@10 for {args.patience} epochs (best {best:.3f} at epoch {best_epoch})", flush=True)
            break


if __name__ == "__main__":
    main()
