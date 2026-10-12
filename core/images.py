"""Download listing photos from the iFood image host given in the assignment brief.

    .venv/bin/python -m core.images [--human-only]

Only the first image of each listing is fetched (the `t_low` size, about 15 KB), politely with
four connections, into .demo-cache/images/<sha1 of url>.jpg. Existing files are skipped, so the
command can be re-run. --human-only fetches just the listings in the human-labelled pairs.
"""

from __future__ import annotations

import argparse
import hashlib
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import httpx

from core.common import HUMAN, ROOT, load_corpus, read_jsonl

IMAGES = ROOT / ".demo-cache" / "images"


def image_path(url: str) -> Path:
    return IMAGES / f"{hashlib.sha1(url.encode()).hexdigest()}.jpg"


def download(urls: list[str]) -> dict[str, int]:
    IMAGES.mkdir(parents=True, exist_ok=True)
    counts = {"downloaded": 0, "cached": 0, "failed": 0}
    with httpx.Client(timeout=30.0, follow_redirects=True) as client:
        def fetch(url: str) -> str:
            path = image_path(url)
            if path.is_file() and path.stat().st_size > 0:
                return "cached"
            try:
                response = client.get(url)
                response.raise_for_status()
                path.write_bytes(response.content)
                return "downloaded"
            except httpx.HTTPError:
                return "failed"

        with ThreadPoolExecutor(4) as pool:
            for outcome in pool.map(fetch, urls):
                counts[outcome] += 1
    return counts


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--human-only", action="store_true")
    args = parser.parse_args()
    corpus = load_corpus()
    wanted = {r["document_id"] for r in read_jsonl(HUMAN)} if args.human_only else None
    urls = sorted({item["image_url"] for item in corpus.items
                   if item["image_url"] and (wanted is None or item["document_id"] in wanted)})
    print(f"{len(urls)} image URLs", flush=True)
    print(download(urls))


if __name__ == "__main__":
    main()
