"""Download the pinned public models for local, repeatable inference.

    .venv/bin/python core/download_model.py [--skip-ollama]

Hugging Face models are pinned to exact revisions. Ollama models are pulled by
tag; the evaluation records each tag's digest. Assignment records are never
uploaded anywhere. The generative-retrieval backbone (T5Gemma 2) is gated: accept
the Gemma terms on its Hugging Face page and provide a read token through
HF_TOKEN or HF_TOKEN_PATH; only that download uses the token.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys

from huggingface_hub import HfApi

ROOT = Path(__file__).resolve().parents[1]
MODELS_DIR = ROOT / ".demo-cache" / "models"
HF_MODELS = {
    "multilingual-e5-small": ("intfloat/multilingual-e5-small", "614241f622f53c4eeff9890bdc4f31cfecc418b3"),
    "harrier-oss-v1-0.6b": ("microsoft/harrier-oss-v1-0.6b", "f9b9dc8d367d443f2479d27aa5d8d2850c0774ee"),
    "llama-nemotron-rerank-1b-v2": ("nvidia/llama-nemotron-rerank-1b-v2", "828765652b05bd439c9789d2a6d093db1caa1443"),
    "t5gemma-2-270m-270m": ("google/t5gemma-2-270m-270m", "7c38f16641f455ef0685b18431faf1b17722d5a1"),
    # Industry-style baseline: an LLM-based (Qwen3-VL) embedder for text, photos, or both.
    "qwen3-vl-embedding-2b": ("Qwen/Qwen3-VL-Embedding-2B", "9f2f7e710d6d81056aa5c0a4f04764fec6bb7bda"),
}
GATED = {"t5gemma-2-270m-270m"}
OLLAMA_MODELS = ["gemma4:e4b-it-q4_K_M", "qwen3.5:9b-q4_K_M"]
_SKIP = {"trainer_state.json", "zero_to_fp32.py"}


def download_hf(name: str, repo: str, revision: str) -> None:
    destination = MODELS_DIR / name
    files = HfApi(token=False).list_repo_files(repo, revision=revision)
    selected = [
        file for file in files
        if file not in _SKIP and ("/" not in file or file.startswith("1_Pooling/"))
        and (file.endswith((".json", ".model", ".txt", ".jinja", ".py")) or file in ("model.safetensors", "LICENSE"))
    ]
    if "model.safetensors" not in selected:
        raise SystemExit(f"{repo} has no expected safetensors weights at {revision}.")
    env = dict(os.environ)
    env["HF_HOME"] = str(ROOT / ".demo-cache" / "huggingface")
    if name not in GATED:
        env["HF_HUB_DISABLE_IMPLICIT_TOKEN"] = "1"  # public models are downloaded anonymously
    env["HF_HUB_DISABLE_TELEMETRY"] = "1"
    env["HF_HUB_DISABLE_PROGRESS_BARS"] = "1"
    cli = Path(sys.executable).parent / "hf"
    print(f"Downloading {repo} at {revision}", flush=True)
    subprocess.run(
        [str(cli), "download", repo, *selected, "--revision", revision,
         "--local-dir", str(destination), "--max-workers", "3", "--quiet"],
        env=env, check=True,
    )
    (destination / "prosus-model.json").write_text(json.dumps({"repo_id": repo, "revision": revision}, indent=2) + "\n")
    print(f"Local model ready: {destination}", flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--skip-ollama", action="store_true")
    parser.add_argument("--only", nargs="*", choices=sorted(HF_MODELS), help="download just these Hugging Face models")
    args = parser.parse_args()
    for name, (repo, revision) in HF_MODELS.items():
        if args.only is None or name in args.only:
            download_hf(name, repo, revision)
    if args.only is not None:
        return
    if args.skip_ollama:
        return
    ollama = shutil.which("ollama")
    if not ollama:
        raise SystemExit("Install Ollama (for example `brew install ollama`), start `ollama serve`, and run again.")
    for tag in OLLAMA_MODELS:
        print(f"Pulling {tag} with Ollama", flush=True)
        subprocess.run([ollama, "pull", tag], check=True)


if __name__ == "__main__":
    main()
