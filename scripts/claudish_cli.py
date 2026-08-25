#!/usr/bin/env python3
"""Run text through the trained Claudish adapter (Q4 GGUF, local llama.cpp).

Defaults to Claudish -> plain English. Reads text from the argument, or stdin
when no argument is given. First run downloads the ~4.4 GB GGUF from the Hub.

Examples:
    claudish_cli.py "The failure surface here is load-bearing."
    claudish_cli.py --to-claudish "The tests failed because the DB wasn't closed."
    echo "some text" | claudish_cli.py
"""
import argparse
import os
import sys

MODEL_REPO = "adamrotmil/claudish-style-adapter"
MODEL_FILE = "gguf/claudish-style-adapter-Q4_K_M.gguf"

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


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("text", nargs="?", help="text to restyle (or pipe via stdin)")
    parser.add_argument("--to-claudish", action="store_true",
                        help="English -> Claudish (default is Claudish -> English)")
    parser.add_argument("--max-tokens", type=int, default=1024)
    args = parser.parse_args()

    text = args.text if args.text is not None else sys.stdin.read()
    text = text.strip()
    if not text:
        sys.exit("no input text")

    from huggingface_hub import hf_hub_download
    from llama_cpp import Llama

    model_path = hf_hub_download(MODEL_REPO, MODEL_FILE)
    llm = Llama(model_path=model_path, n_ctx=2048,
                n_threads=os.cpu_count(), verbose=False)

    direction = "to_claudish" if args.to_claudish else "to_english"
    prompt = PROMPT_TEMPLATE.format(instruction=INSTRUCTIONS[direction], input=text)
    out = llm(prompt, max_tokens=args.max_tokens, temperature=0.0, repeat_penalty=1.1,
              stop=["### Instruction:", "### Input:"])
    print(out["choices"][0]["text"].strip())


if __name__ == "__main__":
    main()
