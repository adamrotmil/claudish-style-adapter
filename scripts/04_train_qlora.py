#!/usr/bin/env python3
"""Phase 6: fine-tune a QLoRA style adapter on the Claudish SFT dataset.

Uses TRL's SFTTrainer with PEFT LoRA on a 4-bit NF4 quantized base model.
Defaults are tuned for style transfer on a single 24 GB GPU.

Example:
    python scripts/04_train_qlora.py \
        --base-model Qwen/Qwen2.5-7B-Instruct \
        --data-dir data/sft \
        --out-dir outputs/claudish-lora
"""
import argparse


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-model", default="Qwen/Qwen2.5-7B-Instruct")
    parser.add_argument("--data-dir", default="data/sft")
    parser.add_argument("--out-dir", default="outputs/claudish-lora")
    parser.add_argument("--epochs", type=float, default=2.0)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--lora-r", type=int, default=32)
    parser.add_argument("--lora-alpha", type=int, default=32)
    parser.add_argument("--lora-dropout", type=float, default=0.05)
    parser.add_argument("--max-length", type=int, default=1024)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--grad-accum", type=int, default=4)
    parser.add_argument("--no-4bit", action="store_true",
                        help="disable 4-bit quantization (needs much more VRAM)")
    args = parser.parse_args()

    import torch
    from datasets import load_dataset
    from peft import LoraConfig
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
    from trl import SFTConfig, SFTTrainer

    dataset = load_dataset(
        "json",
        data_files={
            "train": f"{args.data_dir}/train.jsonl",
            "val": f"{args.data_dir}/val.jsonl",
        },
    )
    # SFTTrainer applies completion-only loss to prompt/completion columns;
    # drop everything else so no stray columns reach the collator.
    dataset = dataset.select_columns(["prompt", "completion"])

    quant_config = None
    if not args.no_4bit:
        quant_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
            bnb_4bit_compute_dtype=torch.bfloat16,
        )

    model = AutoModelForCausalLM.from_pretrained(
        args.base_model,
        quantization_config=quant_config,
        torch_dtype=torch.bfloat16,
        device_map="auto",
    )
    tokenizer = AutoTokenizer.from_pretrained(args.base_model)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    peft_config = LoraConfig(
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                        "gate_proj", "up_proj", "down_proj"],
    )

    sft_config = SFTConfig(
        output_dir=args.out_dir,
        num_train_epochs=args.epochs,
        learning_rate=args.lr,
        per_device_train_batch_size=args.batch_size,
        gradient_accumulation_steps=args.grad_accum,
        max_length=args.max_length,
        lr_scheduler_type="cosine",
        warmup_ratio=0.03,
        logging_steps=20,
        eval_strategy="steps",
        eval_steps=200,
        save_strategy="epoch",
        bf16=True,
        gradient_checkpointing=True,
        report_to="none",
    )

    trainer = SFTTrainer(
        model=model,
        args=sft_config,
        train_dataset=dataset["train"],
        eval_dataset=dataset["val"],
        peft_config=peft_config,
        processing_class=tokenizer,
    )
    trainer.train()
    trainer.save_model(args.out_dir)
    tokenizer.save_pretrained(args.out_dir)
    print(f"adapter saved to {args.out_dir}")


if __name__ == "__main__":
    main()
