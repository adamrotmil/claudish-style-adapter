"""Hugging Face Space demo for the Claudish style adapter.

Runs the merged model as a Q4_K_M GGUF via llama.cpp so it fits on free CPU
hardware. For faster local inference, grab the same GGUF from the model repo.
"""
import os

import gradio as gr
from huggingface_hub import hf_hub_download
from llama_cpp import Llama

MODEL_REPO = "adamrotmil/claudish-style-adapter"
MODEL_FILE = "gguf/claudish-style-adapter-Q4_K_M.gguf"

INSTRUCTIONS = {
    "English → Claudish": "Rewrite the following text in Claudish style while preserving all facts and meaning.",
    "Claudish → English": "Rewrite the following Claudish text into plain, direct English while preserving all facts and meaning.",
}

PROMPT_TEMPLATE = """### Instruction:
{instruction}

### Input:
{input}

### Response:
"""

print("downloading model ...")
model_path = hf_hub_download(MODEL_REPO, MODEL_FILE)
llm = Llama(model_path=model_path, n_ctx=2048,
            n_threads=os.cpu_count(), verbose=False)


def run(text: str, direction: str) -> str:
    text = (text or "").strip()
    if not text:
        return ""
    prompt = PROMPT_TEMPLATE.format(
        instruction=INSTRUCTIONS[direction], input=text)
    out = llm(prompt, max_tokens=768, temperature=0.0, repeat_penalty=1.1,
              stop=["### Instruction:", "### Input:"])
    return out["choices"][0]["text"].strip()


demo = gr.Interface(
    fn=run,
    inputs=[
        gr.Textbox(lines=6, label="Input text",
                   placeholder="Paste a sentence or paragraph ..."),
        gr.Radio(list(INSTRUCTIONS), value="English → Claudish",
                 label="Direction"),
    ],
    outputs=gr.Textbox(lines=6, label="Rewritten"),
    title="Claudish Style Adapter",
    description=(
        "Rewrites text between plain English and **Claudish** — the characteristic prose "
        "style of Claude / Claude Code — while preserving facts and meaning. "
        "A LoRA fine-tune of Qwen2.5-7B-Instruct trained on pairs from the official "
        "[Claudish translator](https://programasweights.com/claudish); running here as a "
        "4-bit GGUF on free CPU hardware, so expect ~30–60 s per rewrite. "
        "Works best on sentence-to-paragraph inputs. "
        f"[Model]( https://huggingface.co/{MODEL_REPO}) · "
        "[Code](https://github.com/adamrotmil/claudish-style-adapter)"
    ),
    examples=[
        ["The tests failed because the database connection wasn't closed.",
         "English → Claudish"],
        ["The failure surface here is load-bearing: the tests fail because the "
         "connection lifecycle has a clean boundary that the code does not respect.",
         "Claudish → English"],
    ],
    cache_examples=False,
)

demo.launch()
