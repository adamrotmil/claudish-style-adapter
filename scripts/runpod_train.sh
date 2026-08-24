#!/usr/bin/env bash
# One-shot training runner for a RunPod GPU pod (24 GB VRAM, PyTorch CUDA template).
# Run from the repo root on the pod, after data/sft/ has been copied up:
#   bash scripts/runpod_train.sh
# Extra args are passed through to 04_train_qlora.py, e.g.:
#   bash scripts/runpod_train.sh --base-model meta-llama/Llama-3.1-8B-Instruct
set -euo pipefail

if [ ! -f data/sft/train.jsonl ]; then
  echo "error: data/sft/train.jsonl not found — copy the formatted dataset up first" >&2
  exit 1
fi

pip install -r requirements-train.txt

python scripts/04_train_qlora.py \
  --base-model "${BASE_MODEL:-Qwen/Qwen2.5-7B-Instruct}" \
  --data-dir data/sft \
  --out-dir outputs/claudish-lora \
  "$@"

python scripts/05_evaluate.py --adapter outputs/claudish-lora --data-dir data/sft --n 100
