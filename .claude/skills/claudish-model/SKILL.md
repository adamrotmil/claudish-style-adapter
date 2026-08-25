---
name: claudish-model
description: Run text through Adam's trained Claudish adapter model (the v3 GGUF, locally on this Mac) — the actual fine-tuned weights, not Claude restyling. Use when the user asks to run text "through the model", "through the adapter", "through my model", or wants to compare the trained adapter's output against Claude's own restyling.
---

# Claudish adapter — local model inference

Runs the user's trained model (adamrotmil/claudish-style-adapter, Q4 GGUF via
llama.cpp/Metal). This is the REAL adapter output, imperfections and all — never
"improve" or correct it; the point is to see what the model actually does.

## How to run

Claudish → plain English (default):

```bash
/Users/adamrotmil/dev/claudish-style-adapter/.venv/bin/python /Users/adamrotmil/dev/claudish-style-adapter/scripts/claudish_cli.py "TEXT HERE"
```

English → Claudish: add `--to-claudish`.

For multi-line text, pipe via stdin instead of an argument. First-ever run downloads
~4.4 GB from Hugging Face; after that it loads in a few seconds and generates at
Metal speed.

## Output

Return the model's output verbatim, clearly labeled as the adapter's output. If the
user wants a comparison, you may add your own restyling per the `claudish-style`
skill — labeled separately — but never blend the two.
