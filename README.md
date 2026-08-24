# Claudish Style Adapter

A lightweight, open-source English ↔ Claudish style adapter: a LoRA fine-tune of a small
open-source model that rewrites text into (or out of) "Claudish" — the characteristic prose
style of Claude and Claude Code — while preserving all facts and meaning.

The pipeline uses the official bidirectional Claudish translator from
[ProgramAsWeights](https://programasweights.com/claudish) to generate a large parallel
dataset, fine-tunes a base model with QLoRA, and publishes the resulting weights on
Hugging Face so anyone can drop the adapter in as a surface-layer style rewriter on top of
any underlying model (Claude, GPT, Grok, local models, ...).

## Pipeline overview

```
seeds (English text)                          scripts/01_collect_seeds.py
        │
        ▼
parallel pairs (English ↔ Claudish)           scripts/02_generate_pairs.py
        │   uses the official translator (programasweights, runs locally via llama.cpp)
        ▼
instruction-formatted train/val JSONL         scripts/03_format_dataset.py
        │   both directions mixed in one multi-task dataset
        ▼
LoRA adapter (QLoRA, 4-bit)                   scripts/04_train_qlora.py
        │
        ▼
evaluation report                             scripts/05_evaluate.py
        │
        ▼
Hugging Face Hub upload + model card          scripts/06_publish.py
```

## Quick start

### 0. Install

Two requirement sets, because data generation and training have very different needs:

```bash
# Data generation (CPU is fine; downloads the translator weights on first use, ~600 MB)
pip install -r requirements.txt --extra-index-url https://pypi.programasweights.com/simple/

# Training + evaluation + publishing (needs a CUDA GPU, ~24 GB VRAM for a 7–8B base)
pip install -r requirements-train.txt
```

### 1. Collect seed English text

```bash
python scripts/01_collect_seeds.py --target 20000 --out data/english_seeds.txt
```

Pulls diverse English sentences/paragraphs from public instruction datasets
(Alpaca, Dolly-15k) and mixes in any local text you provide via `--extra-file`
(e.g. exported chat transcripts, one text per line). Deduplicates and length-filters.

### 2. Generate parallel pairs with the official translator

```bash
python scripts/02_generate_pairs.py --seeds data/english_seeds.txt --out data/claudish_pairs.jsonl
```

Runs every seed through `to_claudish`, optionally back through `to_english`
(round-trip, on by default — used for quality filtering and as extra training signal),
and writes clean JSONL pairs. The script is resumable: re-running it skips seeds that
are already in the output file, so it is safe to interrupt. The translator runs locally
on CPU; expect a few seconds per seed, so for 10k+ seeds run it on a beefy machine or
shard the seed file across processes with `--shard i/n`.

Target size: 10k–30k pairs. A committed sample lives in `data/sample/`.

### 3. Format for fine-tuning

```bash
python scripts/03_format_dataset.py --pairs data/claudish_pairs.jsonl --out-dir data/sft
```

Produces `train.jsonl` / `val.jsonl` (95/5 split) with `prompt`/`completion` columns in the
Alpaca-style instruction format, mixing both directions (English → Claudish and
Claudish → English) in one multi-task dataset.

### 4. Train the QLoRA adapter

```bash
python scripts/04_train_qlora.py \
  --base-model Qwen/Qwen2.5-7B-Instruct \
  --data-dir data/sft \
  --out-dir outputs/claudish-lora
```

Defaults follow settings that work well for style transfer: LoRA rank 32, alpha 32,
all linear projection modules, 2 epochs, lr 1e-4, max length 1024, 4-bit NF4 quantization.
Works on a single 24 GB GPU (or Colab/RunPod). `--base-model meta-llama/Llama-3.1-8B-Instruct`
is the other recommended starting point.

### 5. Evaluate

```bash
python scripts/05_evaluate.py --adapter outputs/claudish-lora --data-dir data/sft --n 100
```

Reports embedding similarity between model output and the reference translation, plus
meaning-preservation similarity between input and output, on held-out validation examples.
With `ANTHROPIC_API_KEY` set and `--judge`, it additionally asks Claude to score each output
for "Claudishness" and faithfulness (no invented facts).

### 6. Publish to Hugging Face

```bash
huggingface-cli login
python scripts/06_publish.py --adapter outputs/claudish-lora --repo YourUsername/claudish-style-adapter
# add --merge to also push a full merged model instead of just the PEFT adapter
```

Uploads the adapter (small download, recommended) or a merged model, the tokenizer, and a
generated model card.

### 7. Use it

```python
from claudish_adapter import rewrite

rewrite("The tests failed because the DB connection wasn't closed.", direction="to_claudish")
rewrite("The failure surface here is load-bearing...", direction="to_english")
```

Or run the Gradio demo:

```bash
python demo/app.py --adapter YourUsername/claudish-style-adapter
```

## The lean alternative: a style guide instead of weights

[`docs/claudish-style.md`](docs/claudish-style.md) is a multi-shot style reference —
rules, signature moves, worked examples, and diagnosed failure cases. Point any capable
model at it ("restyle this text per the guide") and you get higher-quality translations
than the adapter, with the model's intelligence intact. The adapter's advantage is that
it runs anywhere without an API; the guide's advantage is quality. They ship together on
purpose.

## Repository layout

| Path | Purpose |
|---|---|
| `scripts/01_collect_seeds.py` | Phase 2 — collect/dedupe seed English text |
| `scripts/02_generate_pairs.py` | Phase 3 — batch-generate parallel pairs with the official translator |
| `scripts/03_format_dataset.py` | Phase 4 — instruction formatting + train/val split |
| `scripts/04_train_qlora.py` | Phase 6 — QLoRA fine-tuning (TRL + PEFT) |
| `scripts/05_evaluate.py` | Phase 7 — automatic evaluation (+ optional Claude judge) |
| `scripts/06_publish.py` | Phase 8 — merge/upload weights + model card to HF Hub |
| `claudish_adapter/` | Phase 9 — tiny Python client (`rewrite()`) |
| `demo/app.py` | Phase 9 — Gradio demo |
| `data/sample/` | Small committed sample of generated pairs |
| `docs/claudish-spec-notes.md` | Notes on the Claudish style, from the official specs |

## Notes

- The translator function IDs are `ca9d5165b6c8e6615529` (English → Claudish) and
  `e469f61ccab2699fbd51` (Claudish → English), from
  [programasweights/claudish](https://github.com/programasweights/claudish).
- The adapter is a *surface-layer* style rewriter: it must preserve facts, certainty, and
  implications, and never invent content. The evaluation step checks this explicitly.
- Train both directions in one adapter (default) or pass `--direction to_claudish` /
  `--direction to_english` to `03_format_dataset.py` to build single-direction datasets.
