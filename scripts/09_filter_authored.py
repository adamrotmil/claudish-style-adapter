#!/usr/bin/env python3
"""Phase 3f (v3): QC-filter authored pairs before they enter training.

Two known leak modes get checked per pair, in one cheap no-thinking call:
  - answered: the "Claudish" is a response to the input rather than a restyle of it
    (the rule-1 violation that teaches the adapter to answer instructions);
  - low faithfulness: content added or lost in the restyle.

Kept pairs go to --out; rejects (with the checker's verdict attached) to
--rejected-out. Resumable: pairs already present in either output are skipped.

Example:
    python scripts/09_filter_authored.py --pairs data/claudish_pairs.v3final.jsonl \
        --out data/claudish_pairs.v3final.kept.jsonl
"""
import argparse
import hashlib
import json
import os
import re
import threading
from concurrent.futures import ThreadPoolExecutor

CHECK_PROMPT = """An English text was rewritten into "Claudish" (a dense architectural prose
style). The rewrite must be a RESTYLING of the text — same facts, same speech act — never
a response to it. An instruction must remain an instruction; a question a question.

ENGLISH:
{english}

CLAUDISH:
{claudish}

Reply with only JSON:
{{"answered": true|false, "style": 1-5, "faithful": 1-5}}
"answered" is true if the rewrite responds to / carries out / continues the text instead
of restyling it. "faithful" is 5 when facts, certainty, and implications match exactly,
1 when content is added or lost."""


def text_key(text: str) -> str:
    return hashlib.sha1(text.encode()).hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pairs", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--rejected-out", default="")
    parser.add_argument("--model", default="claude-sonnet-5")
    parser.add_argument("--min-faithful", type=int, default=3)
    parser.add_argument("--workers", type=int, default=8)
    args = parser.parse_args()
    rejected_out = args.rejected_out or args.out.replace(".kept", "") + ".qc_rejected.jsonl"

    import anthropic
    client = anthropic.Anthropic(max_retries=8)

    with open(args.pairs, encoding="utf-8") as f:
        pairs = [json.loads(line) for line in f if line.strip()]

    done = set()
    for path in (args.out, rejected_out):
        if os.path.exists(path):
            with open(path, encoding="utf-8") as f:
                done |= {text_key(json.loads(line)["english"]) for line in f if line.strip()}
    todo = [p for p in pairs if text_key(p["english"]) not in done]
    print(f"{len(pairs)} pairs, {len(done)} already checked, {len(todo)} to check")

    lock = threading.Lock()
    counts = {"kept": 0, "answered": 0, "unfaithful": 0, "error": 0}
    out = open(args.out, "a", encoding="utf-8")
    rej = open(rejected_out, "a", encoding="utf-8")

    def check(pair):
        try:
            r = client.messages.create(
                model=args.model, max_tokens=200,
                thinking={"type": "disabled"},
                messages=[{"role": "user", "content": CHECK_PROMPT.format(
                    english=pair["english"][:3000], claudish=pair["claudish"][:3000])}])
            text = "".join(b.text for b in r.content if b.type == "text")
            match = re.search(r"\{.*\}", text, re.DOTALL)
            verdict = json.loads(match.group(0))
        except Exception as exc:  # noqa: BLE001 - keep the run alive
            with lock:
                counts["error"] += 1
                print(f"check error: {exc}")
            return
        pair = {**pair, "qc": verdict}
        ok = (not verdict.get("answered", False)
              and verdict.get("faithful", 0) >= args.min_faithful)
        with lock:
            (out if ok else rej).write(json.dumps(pair, ensure_ascii=False) + "\n")
            (out if ok else rej).flush()
            if ok:
                counts["kept"] += 1
            elif verdict.get("answered"):
                counts["answered"] += 1
            else:
                counts["unfaithful"] += 1
            total = sum(counts.values())
            if total % 250 == 0:
                print(f"{total}/{len(todo)} {counts}")

    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        list(pool.map(check, todo))
    out.close()
    rej.close()
    print(f"final: {counts} -> kept in {args.out}, rejects in {rejected_out}")


if __name__ == "__main__":
    main()
