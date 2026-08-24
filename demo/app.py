#!/usr/bin/env python3
"""Phase 9: Gradio demo for the Claudish style adapter.

Example:
    python demo/app.py --adapter outputs/claudish-lora
    python demo/app.py --adapter YourUsername/claudish-style-adapter --share
"""
import argparse
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from claudish_adapter import rewrite  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--adapter", default="outputs/claudish-lora")
    parser.add_argument("--share", action="store_true")
    args = parser.parse_args()

    import gradio as gr

    def run(text: str, direction: str) -> str:
        if not text.strip():
            return ""
        key = "to_claudish" if direction.startswith("English") else "to_english"
        return rewrite(text, direction=key, adapter=args.adapter)

    demo = gr.Interface(
        fn=run,
        inputs=[
            gr.Textbox(lines=6, label="Input text"),
            gr.Radio(["English → Claudish", "Claudish → English"],
                     value="English → Claudish", label="Direction"),
        ],
        outputs=gr.Textbox(lines=6, label="Rewritten"),
        title="Claudish Style Adapter",
        description=(
            "Rewrites text between plain English and Claudish (the characteristic "
            "prose style of Claude / Claude Code) while preserving facts and meaning."
        ),
    )
    demo.launch(share=args.share)


if __name__ == "__main__":
    main()
