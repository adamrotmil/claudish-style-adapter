#!/usr/bin/env python3
"""Phase 4: format parallel pairs into an instruction-style SFT dataset.

Emits prompt/completion JSONL (TRL's SFTTrainer applies completion-only loss
to this format automatically), mixing both directions into one multi-task
dataset by default, with a 95/5 train/validation split.

Example:
    python scripts/03_format_dataset.py --pairs data/claudish_pairs.jsonl --out-dir data/sft
"""
import argparse
import json
import os
import random

INSTRUCTIONS = {
    "to_claudish": "Rewrite the following text in Claudish style while preserving all facts and meaning.",
    "to_english": "Rewrite the following Claudish text into plain, direct English while preserving all facts and meaning.",
}

PROMPT_TEMPLATE = """### Instruction:
{instruction}

### Input:
{input}

### Response:
"""


def make_example(direction: str, source: str, target: str) -> dict:
    return {
        "prompt": PROMPT_TEMPLATE.format(instruction=INSTRUCTIONS[direction], input=source),
        "completion": target,
        "direction": direction,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pairs", default="data/claudish_pairs.jsonl")
    parser.add_argument("--out-dir", default="data/sft")
    parser.add_argument("--direction", choices=["both", "to_claudish", "to_english"],
                        default="both")
    parser.add_argument("--val-fraction", type=float, default=0.05)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    examples = []
    with open(args.pairs, encoding="utf-8") as f:
        for line in f:
            pair = json.loads(line)
            eng, claud = pair["english"], pair["claudish"]
            if args.direction in ("both", "to_claudish"):
                examples.append(make_example("to_claudish", eng, claud))
            if args.direction in ("both", "to_english"):
                examples.append(make_example("to_english", claud, eng))

    random.Random(args.seed).shuffle(examples)
    n_val = max(1, int(len(examples) * args.val_fraction))
    splits = {"val": examples[:n_val], "train": examples[n_val:]}

    os.makedirs(args.out_dir, exist_ok=True)
    for name, rows in splits.items():
        path = os.path.join(args.out_dir, f"{name}.jsonl")
        with open(path, "w", encoding="utf-8") as f:
            for row in rows:
                f.write(json.dumps(row, ensure_ascii=False) + "\n")
        print(f"wrote {len(rows)} examples to {path}")


if __name__ == "__main__":
    main()
