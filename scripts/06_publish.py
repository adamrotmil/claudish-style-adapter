#!/usr/bin/env python3
"""Phase 8: publish the adapter (or a merged model) to the Hugging Face Hub.

By default pushes the small PEFT adapter plus tokenizer and a generated model
card. Pass --merge to merge the LoRA into the base model and push full weights
instead (much larger upload, but usable without PEFT).

Run `huggingface-cli login` first.

Example:
    python scripts/06_publish.py --adapter outputs/claudish-lora \
        --repo YourUsername/claudish-style-adapter
"""
import argparse

MODEL_CARD = """---
license: apache-2.0
base_model: {base_model}
tags:
  - style-transfer
  - claudish
  - lora
  - text2text
language:
  - en
---

# Claudish Style Adapter

A {kind} that rewrites text between plain English and **Claudish** — the characteristic
prose style of Claude and Claude Code — while preserving all facts and meaning. Use it as a
surface-layer style rewriter on top of any underlying model (Claude, GPT, Grok, local
models, ...).

## How it was trained

Parallel data was generated with the official bidirectional Claudish translator from
[ProgramAsWeights](https://programasweights.com/claudish) (function ids
`ca9d5165b6c8e6615529` and `e469f61ccab2699fbd51`). **v2** (current): 32k seeds →
21.8k pairs surviving an embedding-based meaning-preservation filter, plus 5k synthetic
multi-paragraph pairs → 53.6k instruction examples (both directions), fine-tuned on
`{base_model}` with LoRA (bf16, r=32, all linear projections, 2 epochs, max length
2048). Pipeline code: [claudish-style-adapter](https://github.com/{gh_repo}).

## Evaluation (v2, held out; Claude-judged scores are 1–5)

| Slice | Direction | Ref sim | Meaning | Judge: style | Judge: faithful |
|---|---|---|---|---|---|
| standard | → Claudish | 0.86 | 0.83 | 1.7 | 2.9 |
| standard | → English | 0.90 | 0.86 | 4.0 | 3.2 |
| long (>800 chars) | → Claudish | 0.88 | 0.78 | 2.0 | 1.4 |
| long (>800 chars) | → English | 0.90 | 0.80 | 3.8 | 1.9 |

v2's headline fix over v1: long inputs no longer degenerate (v1 collapsed to ~0.21×
input length with 0.48 reference similarity on the long slice; v2 holds 0.68× and 0.88).

## Honest assessment

This adapter is a distillation of the official translator, and it inherits that
teacher's ceiling — the Claude-judged scores above are the candid measure. The
Claudish → English direction is usable; the English → Claudish direction reads as
imitation-Claudish (judged 1.7–2.0/5 for style) and can drift or, on long inputs,
append unrelated content (an artifact of the synthetic long-pair construction). For
quality-critical restyling, use the
[style guide](https://github.com/{gh_repo}/blob/main/docs/claudish-style.md) with a
capable instruction-following model instead — it ships in the same repo for exactly
this reason. A v3 trained on higher-quality pairs is planned.

## Usage

```python
from peft import AutoPeftModelForCausalLM
from transformers import AutoTokenizer

model = AutoPeftModelForCausalLM.from_pretrained("{repo}", device_map="auto")
tokenizer = AutoTokenizer.from_pretrained("{repo}")

PROMPT = '''### Instruction:
Rewrite the following text in Claudish style while preserving all facts and meaning.

### Input:
%s

### Response:
'''

inputs = tokenizer(PROMPT % "The tests failed because the DB connection wasn't closed.",
                   return_tensors="pt").to(model.device)
output = model.generate(**inputs, max_new_tokens=512, do_sample=False)
print(tokenizer.decode(output[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True))
```

For the reverse direction, use the instruction: *"Rewrite the following Claudish text into
plain, direct English while preserving all facts and meaning."*

## Intended use & limitations

- Surface-layer style rewriting only: the adapter is trained to preserve facts, certainty,
  and implications, and to never invent content — but verify outputs for high-stakes text.
- English only; not intended for restyling code blocks or structured markup.
- Works best on sentence-to-paragraph inputs (training texts were 40–800 characters).
  The English → Claudish direction can degenerate on much longer inputs — chunk long
  documents into paragraphs first.
- Instruction-shaped inputs ("Classify the following...") may occasionally be *answered*
  rather than restyled.
"""


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--adapter", default="outputs/claudish-lora")
    parser.add_argument("--repo", required=True, help="e.g. YourUsername/claudish-style-adapter")
    parser.add_argument("--merge", action="store_true",
                        help="merge LoRA into the base model and push full weights")
    parser.add_argument("--gh-repo", default="adamrotmil/claudish-style-adapter",
                        help="GitHub repo linked from the model card")
    parser.add_argument("--private", action="store_true")
    args = parser.parse_args()

    import torch
    from huggingface_hub import HfApi
    from peft import AutoPeftModelForCausalLM, PeftConfig
    from transformers import AutoTokenizer

    peft_config = PeftConfig.from_pretrained(args.adapter)
    base_model = peft_config.base_model_name_or_path
    tokenizer = AutoTokenizer.from_pretrained(args.adapter)

    if args.merge:
        model = AutoPeftModelForCausalLM.from_pretrained(
            args.adapter, torch_dtype=torch.bfloat16)
        model = model.merge_and_unload()
        kind = "merged fine-tune"
    else:
        model = AutoPeftModelForCausalLM.from_pretrained(args.adapter)
        kind = "LoRA adapter (PEFT)"

    print(f"pushing {kind} to {args.repo} ...")
    model.push_to_hub(args.repo, private=args.private)
    tokenizer.push_to_hub(args.repo, private=args.private)

    card = MODEL_CARD.format(base_model=base_model, repo=args.repo,
                             gh_repo=args.gh_repo, kind=kind)
    HfApi().upload_file(
        path_or_fileobj=card.encode(),
        path_in_repo="README.md",
        repo_id=args.repo,
        repo_type="model",
    )
    print(f"done: https://huggingface.co/{args.repo}")


if __name__ == "__main__":
    main()
