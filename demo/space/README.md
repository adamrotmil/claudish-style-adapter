---
title: Claudish Style Adapter
emoji: 🎭
colorFrom: purple
colorTo: indigo
sdk: gradio
app_file: app.py
pinned: false
license: apache-2.0
short_description: Rewrite text between plain English and Claudish
models:
  - adamrotmil/claudish-style-adapter
---

# Claudish Style Adapter — demo

Interactive demo for [adamrotmil/claudish-style-adapter](https://huggingface.co/adamrotmil/claudish-style-adapter):
a LoRA fine-tune that rewrites text between plain English and **Claudish** (the
characteristic prose style of Claude / Claude Code) while preserving facts and meaning.

Runs the merged model as a Q4_K_M GGUF via llama.cpp on free CPU hardware — expect
~30–60 s per rewrite. Pipeline code:
[github.com/adamrotmil/claudish-style-adapter](https://github.com/adamrotmil/claudish-style-adapter).
