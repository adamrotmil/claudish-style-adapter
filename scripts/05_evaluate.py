#!/usr/bin/env python3
"""Phase 7: evaluate the trained adapter on held-out validation examples.

Automatic metrics per direction:
  - reference similarity: embedding cosine between model output and the
    reference translation (higher = closer to the official translator);
  - meaning preservation: embedding cosine between the input text and the
    model output (a faithful style rewrite should stay high);
  - length ratio distribution (catches truncation and rambling).

With --judge and ANTHROPIC_API_KEY set, also asks Claude to rate each output
for style fidelity and faithfulness (no invented facts).

Example:
    python scripts/05_evaluate.py --adapter outputs/claudish-lora --data-dir data/sft --n 100
"""
import argparse
import json
import os
import re
import statistics
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from claudish_adapter import rewrite  # noqa: E402

JUDGE_PROMPT = """You are evaluating a style-transfer model that rewrites text between plain \
English and "Claudish" (the characteristic prose style of Claude / Claude Code).

Direction: {direction}

INPUT:
{source}

MODEL OUTPUT:
{output}

Rate the model output on two axes, each 1-5:
- "style": how well the output matches the target style of this direction \
(5 = unmistakably the target style, 1 = indistinguishable from the input style).
- "faithful": whether the output preserves the input's facts, certainty, and \
implications without inventing anything (5 = perfectly faithful, 1 = adds or \
loses substantive content).

Reply with only a JSON object: {{"style": <int>, "faithful": <int>}}"""


def judge_scores(client, direction: str, source: str, output: str) -> dict:
    response = client.messages.create(
        model="claude-opus-5",
        max_tokens=256,
        messages=[{"role": "user", "content": JUDGE_PROMPT.format(
            direction=direction, source=source, output=output)}],
    )
    text = "".join(block.text for block in response.content if block.type == "text")
    match = re.search(r"\{.*\}", text, re.DOTALL)
    return json.loads(match.group(0)) if match else {}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--adapter", default="outputs/claudish-lora")
    parser.add_argument("--data-dir", default="data/sft")
    parser.add_argument("--n", type=int, default=100, help="held-out examples to score")
    parser.add_argument("--judge", action="store_true",
                        help="also score with Claude (needs ANTHROPIC_API_KEY)")
    parser.add_argument("--report", default="outputs/eval_report.json")
    args = parser.parse_args()

    from sentence_transformers import SentenceTransformer
    embedder = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")

    def cosine(a: str, b: str) -> float:
        va, vb = embedder.encode([a, b], normalize_embeddings=True)
        return float(va @ vb)

    client = None
    if args.judge:
        import anthropic
        client = anthropic.Anthropic()

    with open(os.path.join(args.data_dir, "val.jsonl"), encoding="utf-8") as f:
        examples = [json.loads(line) for line in f][: args.n]

    rows = []
    for ex in examples:
        # Recover the raw input text from the formatted prompt.
        source = ex["prompt"].split("### Input:\n", 1)[1].rsplit("\n\n### Response:", 1)[0]
        output = rewrite(source, direction=ex["direction"], adapter=args.adapter,
                         max_new_tokens=1024)
        row = {
            "direction": ex["direction"],
            "source": source,
            "reference": ex["completion"],
            "output": output,
            "ref_similarity": cosine(output, ex["completion"]),
            "meaning_similarity": cosine(output, source),
            "length_ratio": len(output) / max(len(source), 1),
        }
        if client is not None:
            row["judge"] = judge_scores(client, ex["direction"], source, output)
        rows.append(row)
        print(f"[{ex['direction']}] ref={row['ref_similarity']:.3f} "
              f"meaning={row['meaning_similarity']:.3f}")

    summary = {}
    for direction in sorted({r["direction"] for r in rows}):
        subset = [r for r in rows if r["direction"] == direction]
        summary[direction] = {
            "n": len(subset),
            "ref_similarity_mean": statistics.mean(r["ref_similarity"] for r in subset),
            "meaning_similarity_mean": statistics.mean(r["meaning_similarity"] for r in subset),
            "length_ratio_median": statistics.median(r["length_ratio"] for r in subset),
        }
        if args.judge:
            judged = [r["judge"] for r in subset if r.get("judge")]
            if judged:
                summary[direction]["judge_style_mean"] = statistics.mean(
                    j["style"] for j in judged)
                summary[direction]["judge_faithful_mean"] = statistics.mean(
                    j["faithful"] for j in judged)

    os.makedirs(os.path.dirname(args.report) or ".", exist_ok=True)
    with open(args.report, "w", encoding="utf-8") as f:
        json.dump({"summary": summary, "rows": rows}, f, ensure_ascii=False, indent=2)
    print(json.dumps(summary, indent=2))
    print(f"full report: {args.report}")


if __name__ == "__main__":
    main()
