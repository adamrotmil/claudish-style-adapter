#!/usr/bin/env python3
"""Phase 3: batch-generate English ↔ Claudish parallel pairs.

Runs every seed through the official bidirectional Claudish translator
(programasweights, runs locally via llama.cpp) and writes clean JSONL pairs:

    {"english": ..., "claudish": ..., "roundtrip_english": ...}

Resumable: seeds already present in the output file are skipped, so the
script can be interrupted and re-run. For large seed files, shard across
machines/processes with --shard i/n (0-based).

Example:
    python scripts/02_generate_pairs.py --seeds data/english_seeds.txt \
        --out data/claudish_pairs.jsonl
"""
import argparse
import hashlib
import json
import os

from tqdm import tqdm

TO_CLAUDISH_ID = "ca9d5165b6c8e6615529"
TO_ENGLISH_ID = "e469f61ccab2699fbd51"


def text_key(text: str) -> str:
    return hashlib.sha1(text.encode()).hexdigest()


def load_done(path: str) -> set:
    done = set()
    if os.path.exists(path):
        with open(path, encoding="utf-8") as f:
            for line in f:
                try:
                    done.add(text_key(json.loads(line)["english"]))
                except (json.JSONDecodeError, KeyError):
                    continue
    return done


def quality_ok(english: str, claudish: str) -> bool:
    """Reject failed or degenerate translations."""
    if not claudish or len(claudish) < 20:
        return False
    if claudish.strip().lower() == english.strip().lower():
        return False  # no visible stylistic transformation
    ratio = len(claudish) / max(len(english), 1)
    return 0.5 <= ratio <= 3.0


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seeds", default="data/english_seeds.txt")
    parser.add_argument("--out", default="data/claudish_pairs.jsonl")
    parser.add_argument("--limit", type=int, default=0, help="stop after N new pairs (0 = all)")
    parser.add_argument("--no-roundtrip", action="store_true",
                        help="skip translating Claudish back to English")
    parser.add_argument("--shard", default="",
                        help="'i/n' to process only every n-th seed with offset i")
    args = parser.parse_args()

    import programasweights as paw
    to_claudish = paw.function(TO_CLAUDISH_ID)
    to_english = None if args.no_roundtrip else paw.function(TO_ENGLISH_ID)

    with open(args.seeds, encoding="utf-8") as f:
        seeds = [line.strip() for line in f if line.strip()]
    if args.shard:
        i, n = (int(x) for x in args.shard.split("/"))
        seeds = seeds[i::n]

    done = load_done(args.out)
    print(f"{len(seeds)} seeds, {len(done)} already done")

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    written = skipped = 0
    with open(args.out, "a", encoding="utf-8") as out:
        for eng in tqdm(seeds):
            if text_key(eng) in done:
                continue
            claud = (to_claudish(eng) or "").strip()
            if not quality_ok(eng, claud):
                skipped += 1
                continue
            pair = {"english": eng, "claudish": claud}
            if to_english is not None:
                pair["roundtrip_english"] = (to_english(claud) or "").strip()
            out.write(json.dumps(pair, ensure_ascii=False) + "\n")
            out.flush()
            written += 1
            if args.limit and written >= args.limit:
                break

    print(f"wrote {written} new pairs to {args.out} ({skipped} rejected by quality filter)")


if __name__ == "__main__":
    main()
