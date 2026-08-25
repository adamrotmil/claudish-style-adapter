#!/usr/bin/env python3
"""Phase 3d (v3): author high-quality Claudish with Claude instead of the small teacher.

Takes English texts from an existing pair file (the seeds we already curated) and asks
Claude to write the Claudish side per the style guide, replacing the 0.6B translator's
output. This raises the dataset's ceiling from "imitation of a small distilled model"
to "the actual style, written natively".

Resumable like 02: English texts already present in --out are skipped.

Example (pilot):
    ANTHROPIC_API_KEY=... python scripts/07_author_pairs.py \
        --pairs data/claudish_pairs.all_short.jsonl --out data/claudish_pairs.v3.jsonl \
        --n 200 --judge-sample 40
"""
import argparse
import hashlib
import json
import os
import random
import re

STYLE_GUIDE_PATH = os.path.join(os.path.dirname(__file__), "..", "docs", "claudish-style.md")

AUTHOR_INSTRUCTION = """Rewrite the text the user sends into Claudish, exactly per the style
guide above: preserve every fact, instruction, degree of certainty, and implication; use
two or three signature moves; never answer or extend the text; keep roughly comparable
length. Remember the contrast rule: "X, not Y" asserts *not Y*, so only write a contrast
whose negation the input itself states or directly implies — never one that adds an
alternative, constraint, or disambiguation. Prefer one exact metaphor over two
decorative ones. Reply with ONLY the restyled
text."""

JUDGE_PROMPT = """You are evaluating a Claudish restyling (the characteristic prose style of
Claude / Claude Code) of an English text.

ENGLISH:
{english}

CLAUDISH:
{claudish}

Rate 1-5: "style" (5 = unmistakably Claudish) and "faithful" (5 = same facts, certainty,
and implications, nothing added or lost). Reply with only JSON: {{"style": n, "faithful": n}}"""


def text_key(text: str) -> str:
    return hashlib.sha1(text.encode()).hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pairs", default="data/claudish_pairs.all_short.jsonl",
                        help="source of English texts (claudish side is discarded)")
    parser.add_argument("--out", default="data/claudish_pairs.v3.jsonl")
    parser.add_argument("--n", type=int, default=200, help="pairs to author this run")
    parser.add_argument("--model", default="claude-sonnet-5")
    parser.add_argument("--judge-model", default="claude-opus-5")
    parser.add_argument("--judge-sample", type=int, default=0,
                        help="after authoring, judge this many of the new pairs")
    parser.add_argument("--workers", type=int, default=8,
                        help="concurrent authoring requests")
    parser.add_argument("--max-out", type=int, default=1500,
                        help="max output tokens per authored pair")
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    import threading
    from concurrent.futures import ThreadPoolExecutor

    import anthropic
    client = anthropic.Anthropic(max_retries=8)
    guide = open(STYLE_GUIDE_PATH, encoding="utf-8").read()
    # The guide + instruction go in the system block with cache_control, so the
    # large prefix is billed once per 5-minute window, not once per request.
    system = [{"type": "text", "text": guide + "\n\n---\n\n" + AUTHOR_INSTRUCTION,
               "cache_control": {"type": "ephemeral"}}]

    with open(args.pairs, encoding="utf-8") as f:
        texts = [json.loads(line)["english"] for line in f if line.strip()]
    texts = sorted(set(texts))
    random.Random(args.seed).shuffle(texts)

    done = set()
    if os.path.exists(args.out):
        with open(args.out, encoding="utf-8") as f:
            done = {text_key(json.loads(line)["english"]) for line in f if line.strip()}
    todo = [t for t in texts if text_key(t) not in done][: args.n]
    print(f"{len(texts)} candidate texts, {len(done)} already authored, {len(todo)} to author")

    def ask(model, prompt, max_tokens=1500, use_system=False, no_thinking=False):
        response = client.messages.create(
            model=model, max_tokens=max_tokens,
            system=system if use_system else anthropic.NOT_GIVEN,
            thinking={"type": "disabled"} if no_thinking else anthropic.NOT_GIVEN,
            messages=[{"role": "user", "content": prompt}])
        return "".join(b.text for b in response.content if b.type == "text").strip()

    lock = threading.Lock()
    new_pairs = []
    out = open(args.out, "a", encoding="utf-8")

    def author(eng):
        try:
            claud = ask(args.model, eng, max_tokens=args.max_out, use_system=True, no_thinking=True)
        except Exception as exc:  # noqa: BLE001 - keep the run alive
            print(f"error, skipping: {exc}")
            return
        if not claud or len(claud) < 20:
            return
        pair = {"english": eng, "claudish": claud, "author": args.model}
        with lock:
            out.write(json.dumps(pair, ensure_ascii=False) + "\n")
            out.flush()
            new_pairs.append(pair)
            if len(new_pairs) % 100 == 0:
                print(f"{len(new_pairs)}/{len(todo)}")

    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        list(pool.map(author, todo))
    out.close()
    print(f"authored {len(new_pairs)} pairs -> {args.out}")

    if args.judge_sample and new_pairs:
        sample = random.Random(1).sample(new_pairs, min(args.judge_sample, len(new_pairs)))
        styles, faiths = [], []
        for p in sample:
            try:
                text = ask(args.judge_model, JUDGE_PROMPT.format(
                    english=p["english"], claudish=p["claudish"]), max_tokens=800)
                match = re.search(r"\{.*\}", text, re.DOTALL)
                score = json.loads(match.group(0)) if match else {}
                if "style" in score:
                    styles.append(score["style"])
                    faiths.append(score["faithful"])
            except Exception as exc:  # noqa: BLE001
                print(f"judge error: {exc}")
        if styles:
            print(f"judged {len(styles)}: style mean {sum(styles)/len(styles):.2f}, "
                  f"faithful mean {sum(faiths)/len(faiths):.2f}")


if __name__ == "__main__":
    main()
