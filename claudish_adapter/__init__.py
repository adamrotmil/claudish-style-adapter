"""Tiny client for the Claudish style adapter.

Usage:
    from claudish_adapter import rewrite
    rewrite("some text", direction="to_claudish")
    rewrite("some Claudish text", direction="to_english")

The first call loads the model; pass `adapter=` to point at a local adapter
directory or a Hugging Face repo id. If the adapter repo/directory is a PEFT
adapter, its base model is resolved from the adapter config automatically.
"""
from functools import lru_cache

DEFAULT_ADAPTER = "outputs/claudish-lora"

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


@lru_cache(maxsize=2)
def _load(adapter: str):
    import torch
    from peft import AutoPeftModelForCausalLM, PeftConfig
    from transformers import AutoModelForCausalLM, AutoTokenizer

    dtype = torch.bfloat16 if torch.cuda.is_available() else torch.float32
    try:
        PeftConfig.from_pretrained(adapter)
        model = AutoPeftModelForCausalLM.from_pretrained(
            adapter, dtype=dtype, device_map="auto")
    except ValueError:
        # Not a PEFT adapter: assume a merged full model.
        model = AutoModelForCausalLM.from_pretrained(
            adapter, dtype=dtype, device_map="auto")
    tokenizer = AutoTokenizer.from_pretrained(adapter)
    model.eval()
    return model, tokenizer


def rewrite(text: str, direction: str = "to_claudish",
            adapter: str = DEFAULT_ADAPTER, max_new_tokens: int = 512) -> str:
    if direction not in INSTRUCTIONS:
        raise ValueError(f"direction must be one of {sorted(INSTRUCTIONS)}")
    import torch

    model, tokenizer = _load(adapter)
    prompt = PROMPT_TEMPLATE.format(instruction=INSTRUCTIONS[direction], input=text)
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    with torch.no_grad():
        output = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            no_repeat_ngram_size=8,
            repetition_penalty=1.05,
            pad_token_id=tokenizer.pad_token_id or tokenizer.eos_token_id,
        )
    completion = output[0][inputs["input_ids"].shape[1]:]
    return tokenizer.decode(completion, skip_special_tokens=True).strip()
