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

**v3** (current): the Claudish side of every training pair is authored by Claude
Opus per the repo's [style guide](https://github.com/{gh_repo}/blob/main/docs/claudish-style.md),
replacing the earlier distillation of the official
[ProgramAsWeights translator](https://programasweights.com/claudish) (whose pairs judged
2.2/5 style, 2.4/5 faithfulness — the v1/v2 quality ceiling). 10.2k pairs survived a
QC filter (rejecting answered-instead-of-restyled and unfaithful rewrites), including
1.2k coherent multi-paragraph documents built from embedding-clustered related texts.
Fine-tuned on `{base_model}` with LoRA (bf16, r=32, all linear projections, 2 epochs,
max length 2048). The full training set is released as
[adamrotmil/claudish-pairs](https://huggingface.co/datasets/adamrotmil/claudish-pairs)
(CC-BY-NC-4.0, with per-pair QC verdicts). Pipeline code:
[claudish-style-adapter](https://github.com/{gh_repo}).

## Evaluation (v3 held out; Claude-judged scores are 1-5; v2 in parentheses)

| Slice | Direction | Meaning | Judge: style | Judge: faithful |
|---|---|---|---|---|
| standard | → Claudish | 0.85 | 2.9 (1.7) | 3.3 (2.9) |
| standard | → English | 0.84 | 4.8 (4.0) | 3.8 (3.2) |
| long (>800 chars) | → Claudish | 0.88 (0.78) | 3.4 (2.0) | 1.8 (1.4) |
| long (>800 chars) | → English | 0.90 (0.80) | 4.0 (3.8) | 2.3 (1.9) |

## Honest assessment

Every judged metric improved over v2, some sharply — Claudish → English is now genuinely
strong. Two limitations remain. Long-input faithfulness is the weak axis (1.8–2.8/5):
a 7B model restyling multi-paragraph text still drops or substitutes details, so verify
outputs on long documents. And the adapter scores below its own training data (which
judged 3.6/5 style, 3.8/5 faithful) — the usual distillation gap. For quality-critical
restyling, use the style guide with a capable instruction-following model; it ships in
the same repo for exactly this reason.

**Plain English** *(the paragraph above, rendered by this adapter itself — verbatim,
unedited)*:

> All judged metrics improved over v2, some sharply. Claudish → English is now genuinely
> strong. Two limitations remain: 1) Long-input faithfulness (1.8–2.8/5): A 7B model
> restyling multi-paragraph text still drops or substitutes details, so verify outputs on
> long documents; and 2) Adapter scores below its own training data (which judged 3.6/5
> style, 3.8/5 faithful). For quality-critical restyling, use the style guide with a
> capable instruction-following model, which ships in the same repo for this reason.
>
> *(It dropped one phrase — "the usual distillation gap" — which rather proves the point.)*

## Examples (real held-out outputs, with the judge's scores)

**The good** — English → Claudish (style 3, faithful 5):

> *In:* Compare and contrast two potential methods of customer segmentation.
> *Out:* Set two candidate customer segmentation methods against each other: where they
> align, and where they diverge.

**The good** — Claudish → English (style 5, faithful 5):

> *In:* Sort the items along the alphabetical spine.
> *Out:* Organize the items in alphabetical order.

**The bad** — English → Claudish (style 3, faithful 3): the restyle works but the added
imagery ("a seam in the fabric of how we live") edges past what the input states:

> *In:* Six philosophical questions are: - What is the purpose of life? - Are our
> actions predetermined or do we have free will? ...
> *Out:* Six philosophical questions, each one a seam in the fabric of how we live: -
> What is the purpose of life? - Are our actions predetermined, or do we hold free
> will? ...

**The ugly** — a long technical listing (style 2, faithful 1): the prose survives but
the model **silently corrupted numbers** (learning rate 0.001 → 0.01, max epochs
100 → 10). This is the long-input faithfulness limitation in the flesh — never trust
the adapter with data-bearing text you don't verify:

> *In:* Configuration Parameters: - Optimizer: Adam - Loss Function: Binary Cross
> Entropy - Learning rate: 0.001 - Batch size: 32 - Maximum Epochs: 100 ...
> *Out:* Configuration Parameters — the knobs that can be tuned: - Optimizer: Adam -
> Loss Function: Binary Cross Entropy - Learning rate: 0.01 - Batch size: 32 -
> Maximum Epochs: 10 ...

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
- Strongest on sentence-to-paragraph inputs. Long documents no longer degenerate (fixed
  in v2/v3), but their faithfulness scores are the weakest — verify long outputs.
- Instruction-shaped inputs are trained to be restyled, never answered (v3 QC-filtered
  the failure); rare exceptions can still occur.

## Contributing

Improvements are welcome, in either venue:

- **This Hub repo**: open a Discussion or a Pull Request in the Community tab —
  including better eval examples, quantizations, or corrected metadata.
- **[The GitHub repo](https://github.com/{gh_repo})**: the entire pipeline is
  reproducible from scripts `01`–`09` (seed collection → authoring → QC filtering →
  training → judged evaluation → publishing), so a better dataset or training recipe
  can be carried end-to-end and measured with the same judge.

Apache-2.0 all the way down: fine-tunes, quantizations, and derivative adapters are
explicitly encouraged. The most valuable open problem is **long-input faithfulness**
(see the table above); a close second is closing the distillation gap to the training
data. If you train something better, open an issue — happy to link it here.
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
