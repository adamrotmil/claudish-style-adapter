#!/usr/bin/env python3
"""Phase 2: collect diverse seed English text for translation.

Pulls human-written and assistant-written English from public instruction
datasets, mixes in any local text files you provide, deduplicates, and
length-filters. Output is one seed text per line.

Examples:
    python scripts/01_collect_seeds.py --target 20000 --out data/english_seeds.txt
    python scripts/01_collect_seeds.py --target 5000 --extra-file my_chats.txt --no-hf
"""
import argparse
import hashlib
import random
import re


def clean(text: str) -> str:
    text = re.sub(r"\s+", " ", text).strip()
    return text


def usable(text: str, min_chars: int, max_chars: int) -> bool:
    if not (min_chars <= len(text) <= max_chars):
        return False
    # Skip texts that are mostly code or markup: they translate poorly and
    # the adapter should not restyle code anyway.
    if text.count("```") or text.count("</") > 1:
        return False
    letters = sum(c.isalpha() for c in text)
    return letters / max(len(text), 1) > 0.6


def hf_seeds(limit_per_source: int):
    """Yield candidate texts from public instruction datasets."""
    from datasets import load_dataset

    sources = [
        # (dataset, split, fields to extract)
        ("tatsu-lab/alpaca", "train", ["instruction", "output"]),
        # "context" adds declarative wiki-style paragraphs — balances the
        # instruction-heavy fields so the adapter learns to restyle statements,
        # not just imperatives.
        ("databricks/databricks-dolly-15k", "train", ["instruction", "response", "context"]),
    ]
    for name, split, fields in sources:
        try:
            ds = load_dataset(name, split=split)
        except Exception as exc:  # noqa: BLE001 - dataset access is best-effort
            print(f"warning: could not load {name}: {exc}")
            continue
        count = 0
        for row in ds.shuffle(seed=0):
            for field in fields:
                text = clean(str(row.get(field) or ""))
                if text:
                    yield text
                    count += 1
            if count >= limit_per_source:
                break


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", default="data/english_seeds.txt")
    parser.add_argument("--target", type=int, default=20000,
                        help="number of seed texts to collect")
    parser.add_argument("--extra-file", action="append", default=[],
                        help="local text file (one seed per line); repeatable")
    parser.add_argument("--no-hf", action="store_true",
                        help="skip Hugging Face datasets, use only --extra-file")
    parser.add_argument("--min-chars", type=int, default=40)
    parser.add_argument("--max-chars", type=int, default=800)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    candidates = []
    for path in args.extra_file:
        with open(path, encoding="utf-8") as f:
            candidates.extend(clean(line) for line in f)
    if not args.no_hf:
        candidates.extend(hf_seeds(limit_per_source=args.target))

    seen = set()
    seeds = []
    for text in candidates:
        if not usable(text, args.min_chars, args.max_chars):
            continue
        key = hashlib.sha1(text.lower().encode()).hexdigest()
        if key in seen:
            continue
        seen.add(key)
        seeds.append(text)

    random.Random(args.seed).shuffle(seeds)
    seeds = seeds[: args.target]

    import os
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as f:
        f.write("\n".join(seeds) + "\n")
    print(f"wrote {len(seeds)} seeds to {args.out}")


if __name__ == "__main__":
    main()
