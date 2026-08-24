# Training on RunPod

Everything before training runs locally on CPU; only steps 4–6 of the pipeline
(train, evaluate, publish) need the pod.

## Pod choice

- GPU: 24 GB VRAM is enough for the default 7B base with QLoRA — RTX 4090 or
  A5000 are the cheap sweet spot. More VRAM just lets you raise `--batch-size`.
- Template: any recent **PyTorch + CUDA** template (e.g. "RunPod PyTorch 2.x").
- Disk: 60 GB+ container/volume, mostly for the base model download (~16 GB)
  plus checkpoints.

## Steps

1. **Get the code onto the pod** (from the pod's terminal):

   ```bash
   git clone <this-repo-url> claudish-style-adapter && cd claudish-style-adapter
   ```

   Or, with no remote yet, rsync the repo up from the desktop (excluding local
   junk) using the pod's SSH details:

   ```bash
   rsync -av -e "ssh -p <PORT>" --exclude .venv --exclude data --exclude outputs \
     ~/dev/claudish-style-adapter/ root@<POD_IP>:/workspace/claudish-style-adapter/
   ```

2. **Copy the formatted dataset up** (it is gitignored; generated locally by
   `03_format_dataset.py`). From the desktop:

   ```bash
   scp -P <PORT> -r ~/dev/claudish-style-adapter/data/sft root@<POD_IP>:/workspace/claudish-style-adapter/data/
   ```

   `data/sft/` is a few tens of MB — seconds to transfer.

3. **Train + evaluate** (one command, on the pod):

   ```bash
   bash scripts/runpod_train.sh
   ```

   This installs `requirements-train.txt`, runs `04_train_qlora.py` with the
   defaults (Qwen2.5-7B-Instruct, LoRA r=32, 2 epochs, 4-bit NF4), then runs
   `05_evaluate.py` on 100 held-out examples. Rough wall-clock for ~40k SFT
   examples at max_length 1024 on a 4090: several hours — start it under
   `tmux`/`nohup` so an SSH drop doesn't kill it.

   For the optional Claude judge in evaluation, export `ANTHROPIC_API_KEY` on
   the pod and re-run `05_evaluate.py` with `--judge`.

   Note: `04_train_qlora.py` and `05_evaluate.py` are written but were never
   executed on a GPU — expect possible minor API friction with the installed
   TRL/PEFT versions and fix forward (the handoff doc flags this too).

4. **Publish straight from the pod** (small PEFT adapter, recommended):

   ```bash
   huggingface-cli login   # paste an HF write token
   python scripts/06_publish.py --adapter outputs/claudish-lora --repo <hf-username>/claudish-style-adapter
   ```

   Add `--merge` to also push full merged weights.

5. **Copy the adapter back down** (belt-and-braces, from the desktop):

   ```bash
   scp -P <PORT> -r root@<POD_IP>:/workspace/claudish-style-adapter/outputs/claudish-lora ~/dev/claudish-style-adapter/outputs/
   ```

6. Stop the pod — nothing else needs the GPU. The Gradio demo
   (`demo/app.py`) and `rewrite()` client pull the published adapter from HF.

## Sanity bar for evaluation

Meaning-preservation similarity ≳ 0.8, outputs visibly restyled, no invented
facts. If style transfer looks weak, the usual levers are more epochs (2 → 3)
or a bigger dataset; if meaning drifts, lower lr (1e-4 → 5e-5).
