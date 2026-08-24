#!/usr/bin/env python3
"""Phase 3c: synthesize long multi-paragraph pairs from verified short pairs.

The v1 adapter degenerates on inputs much longer than its 40-800 char training
texts. Rather than trusting the small teacher translator on long documents,
this builds long parallel examples by concatenating 2-4 already-filtered short
pairs as paragraphs: the English sides joined make the long English text, the
Claudish sides joined make its translation. Alignment is exact by construction.

Example:
    python scripts/02c_make_long_pairs.py --pairs data/claudish_pairs.filtered.jsonl \
        --out data/claudish_pairs.long.jsonl --n 4000
"""
import argparse
import json
import random


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pairs", default="data/claudish_pairs.filtered.jsonl")
    parser.add_argument("--out", default="data/claudish_pairs.long.jsonl")
    parser.add_argument("--n", type=int, default=4000, help="long pairs to build")
    parser.add_argument("--min-parts", type=int, default=2)
    parser.add_argument("--max-parts", type=int, default=4)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    with open(args.pairs, encoding="utf-8") as f:
        pairs = [json.loads(line) for line in f if line.strip()]

    rng = random.Random(args.seed)
    written = 0
    with open(args.out, "w", encoding="utf-8") as out:
        while written < args.n:
            parts = rng.sample(pairs, rng.randint(args.min_parts, args.max_parts))
            record = {
                "english": "\n\n".join(p["english"] for p in parts),
                "claudish": "\n\n".join(p["claudish"] for p in parts),
                "synthetic_long": True,
            }
            out.write(json.dumps(record, ensure_ascii=False) + "\n")
            written += 1

    print(f"wrote {written} long pairs to {args.out}")


if __name__ == "__main__":
    main()
