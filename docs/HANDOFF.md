# Handoff notes — Claudish style adapter

State handoff for continuing this project in a new Claude session (Claude Desktop +
RunPod GPU). Read `../README.md` first for the pipeline overview; this file covers
what is already done, environment gotchas, and the exact next steps.

## Where things stand (2026-08-24, standalone repo)

This project now lives in its own repo at `~/dev/claudish-style-adapter` (imported from
the `claude/claudish-style-adapter-bzqik0` branch of `adamrotmil/portfolio`, which still
holds the original history). Everything that used to be under `claudish-adapter/` is now
at the repo root; no GitHub remote yet.

Since the move, on the desktop (Apple Silicon, 10 cores, 32 GB):

- Environment: `uv venv --python 3.12 .venv`, then
  `uv pip install -p .venv/bin/python -r requirements.txt --extra-index-url https://pypi.programasweights.com/simple/`.
  Both translator directions verified working (llama.cpp uses Metal here, ~2 s/call).
- Seeds: `data/english_seeds.txt` — 20,000 seeds from Alpaca + Dolly.
- Pairs: DONE — generated as 4 parallel shards (~2.5 h wall clock), merged to
  `data/claudish_pairs.jsonl`: 17,768 raw pairs (~11% of translations rejected
  by the script's built-in degeneracy filter).
- Meaning-preservation filter: DONE — new `scripts/02b_filter_pairs.py` embeds
  pairs with MiniLM and drops those where claudish or roundtrip drifts from the
  source (threshold 0.70/0.70, calibrated by inspecting the borderline band).
  Kept 14,050 pairs (`data/claudish_pairs.filtered.jsonl`); rejects with scores
  in `data/claudish_pairs.rejected.jsonl`.
- SFT split: DONE — `data/sft/train.jsonl` (26,695) + `val.jsonl` (1,405),
  both directions mixed, formatted from the *filtered* pairs.
- RunPod kit: `docs/RUNPOD.md` + `scripts/runpod_train.sh` cover the GPU steps.
  Next action: copy `data/sft/` to the pod and run `scripts/runpod_train.sh`.
- V1 SHIPPED: trained on 3xH100 (12 min, bf16 LoRA DDP), evaluated (ref sim
  0.87/0.93, meaning 0.86/0.87), published at
  https://huggingface.co/adamrotmil/claudish-style-adapter (+ Q4_K_M GGUF in gguf/).
  Known v1 limits: ->Claudish degenerates on inputs >>800 chars; instruction-shaped
  inputs sometimes answered rather than restyled. Anti-repetition decoding added to
  the client as a first mitigation.
- V2 dataset: DONE - 12k new seeds (declarative Dolly context + longer, v1-disjoint)
  -> 9,831 raw -> 7,763 filtered; plus 5,000 synthetic long pairs (02c, 3-6 components
  >=150 chars each; 18% of final pairs now >800 chars, p90 1,449). Final:
  26,813 pairs -> data/sft/ with 50,945 train + 2,681 val examples.
- V2 training: pending a fresh GPU pod. Use `--max-length 2048` (long examples would
  truncate at 1024). Then evaluate - add a long-input eval slice and `--judge` - and
  publish to the same HF repo.
- HF Space demo: written (demo/space/), blocked on HF PRO subscription for Gradio
  hosting; deployable in one command once subscribed.

Done and verified by real runs:

- Full pipeline scaffolding under `claudish-adapter/` (Phases 1–9 of the project plan):
  seed collection, pair generation, dataset formatting, QLoRA training, evaluation,
  HF publishing, `rewrite()` client, Gradio demo.
- The official translator (`programasweights` v0.4.4) installed and working locally:
  `paw.function("ca9d5165b6c8e6615529")` = English → Claudish,
  `paw.function("e469f61ccab2699fbd51")` = Claudish → English.
  First call downloads ~600 MB (Qwen3-0.6B Q6_K GGUF + LoRA adapters) to
  `~/.cache/programasweights/`. Runs on CPU via llama.cpp, ~2 s per translation
  after the prompt prefix is cached.
- A real 40-pair sample generated with `scripts/02_generate_pairs.py` is committed at
  `data/sample/claudish_pairs.sample.jsonl`, and its formatted split at
  `data/sample/sft/` — use these as the reference for expected data shapes.

Not yet done (no GPU in the previous environment):

- Full-scale seed collection and pair generation (target 10k–30k pairs).
- QLoRA training, evaluation, and Hugging Face publishing. The scripts are written but
  **unexercised** — expect possible minor API friction with the installed TRL/PEFT
  versions and fix forward.

## Environment gotchas

1. **Install command** (the translator index is required):
   `pip install -r requirements.txt --extra-index-url https://pypi.programasweights.com/simple/`
2. **llama-cpp-python SIGILL**: on one cloud VM the CPU advertised AVX-512 but faulted
   on it (crash: "Illegal instruction" during decode with a LoRA applied). If that
   happens, rebuild AVX2-only:
   `CMAKE_ARGS="-DGGML_NATIVE=OFF -DGGML_AVX=ON -DGGML_AVX2=ON -DGGML_FMA=ON -DGGML_F16C=ON -DGGML_AVX512=OFF" pip install --force-reinstall --no-cache-dir --no-binary llama-cpp-python llama-cpp-python`
   On a normal desktop or RunPod machine the stock wheel will most likely just work —
   only rebuild if it actually crashes.
3. Pair generation is CPU-bound and embarrassingly parallel: `--shard i/n` on
   `02_generate_pairs.py` splits the seed file across processes/machines, and the
   script is resumable (re-runs skip already-generated pairs).

## Next steps, in order

1. **Seeds**: `python scripts/01_collect_seeds.py --target 20000 --out data/english_seeds.txt`
   (add `--extra-file` with exported chat text for more diversity).
2. **Pairs**: run `02_generate_pairs.py` over the seeds — on a single machine this is
   ~11 h for 20k pairs, so shard it (e.g. 4–8 processes) or run it on a many-core box.
   A RunPod CPU pod works; the GPU is not needed for this step.
3. **Format**: `python scripts/03_format_dataset.py --pairs data/claudish_pairs.jsonl --out-dir data/sft`
4. **Train on RunPod** (24 GB GPU is enough — e.g. RTX 4090/A5000; use a PyTorch CUDA
   template): `pip install -r requirements-train.txt`, then
   `python scripts/04_train_qlora.py --base-model Qwen/Qwen2.5-7B-Instruct --data-dir data/sft --out-dir outputs/claudish-lora`
   Copy `data/sft/` up to the pod; copy `outputs/claudish-lora/` back down (or publish
   straight from the pod).
5. **Evaluate**: `python scripts/05_evaluate.py --adapter outputs/claudish-lora --data-dir data/sft --n 100`
   — add `--judge` with `ANTHROPIC_API_KEY` set for LLM-judged style/faithfulness scores.
   Sanity bar: meaning similarity high (≳0.8), outputs visibly restyled, no invented facts.
6. **Publish**: `huggingface-cli login`, then
   `python scripts/06_publish.py --adapter outputs/claudish-lora --repo <hf-username>/claudish-style-adapter`
   (`--merge` for full merged weights; default pushes the small PEFT adapter).
7. Optional: GGUF/AWQ quantized export, and the Gradio demo (`demo/app.py`).

## Open decisions (defaults chosen so far, all overridable)

- Base model: Qwen/Qwen2.5-7B-Instruct (alt: meta-llama/Llama-3.1-8B-Instruct — needs
  HF gated-repo access).
- One bidirectional adapter (both directions mixed) rather than two single-direction ones.
- Training stack: TRL + PEFT rather than Unsloth. If iteration speed on RunPod matters,
  porting `04_train_qlora.py` to Unsloth is straightforward.
- Dataset size: start with ~20k seeds → ~15–25k usable pairs → ~30–50k SFT examples
  (both directions).
