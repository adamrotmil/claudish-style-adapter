#!/usr/bin/env python3
"""OpenAI-compatible shim that serves the Claudish adapter as a rewrite engine.

Built for the claudish-to-english plugin (CLAUDISH_PROVIDER=openai pointed at
this server), but any /chat/completions client works. The incoming system
prompt is ignored — the rewrite instruction is baked into the adapter — and the
last user message is treated as the text to rewrite into plain English.

Reliability is mechanical, not promised:
  - number guard: every number, URL, and file path in the input must appear
    unchanged in the output; failing outputs are retried (temp 0 -> 0.3 -> 0.6)
    and, if still failing, the ORIGINAL text is returned untouched (fail-open);
  - length guard: inputs too long for the context window pass through unchanged
    rather than being truncated into corruption.

Run:  python scripts/claudish_server.py [--port 8017]
Stdlib + llama_cpp only; no web framework needed.
"""
import argparse
import json
import os
import re
import time
import uuid
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

MODEL_REPO = "adamrotmil/claudish-style-adapter"
MODEL_FILE = "gguf/claudish-style-adapter-Q4_K_M.gguf"

PROMPT_TEMPLATE = """### Instruction:
Rewrite the following Claudish text into plain, direct English while preserving all facts and meaning.

### Input:
{input}

### Response:
"""

MAX_INPUT_CHARS = 4200  # ~1300 tokens; leaves room for output inside n_ctx=2048

GUARD_PATTERN = re.compile(
    r"https?://\S+"            # URLs
    r"|(?:~|\.{0,2}/)[\w./-]+" # file paths
    r"|\d[\d,.:]*\d|\d"        # numbers (incl. decimals, times, versions)
)


def guarded_tokens(text: str) -> list:
    return GUARD_PATTERN.findall(text)


def guard_ok(source: str, output: str) -> bool:
    out = output or ""
    return all(tok in out for tok in guarded_tokens(source))


class Engine:
    def __init__(self):
        from huggingface_hub import hf_hub_download
        from llama_cpp import Llama
        path = hf_hub_download(MODEL_REPO, MODEL_FILE)
        self.llm = Llama(model_path=path, n_ctx=2048,
                         n_threads=os.cpu_count(), verbose=False)

    def rewrite(self, text: str) -> dict:
        text = text.strip()
        if not text:
            return {"text": "", "guard": "empty"}
        if len(text) > MAX_INPUT_CHARS:
            return {"text": text, "guard": "too_long_passthrough"}
        prompt = PROMPT_TEMPLATE.format(input=text)
        for temp in (0.0, 0.3, 0.6):
            out = self.llm(prompt, max_tokens=900, temperature=temp,
                           repeat_penalty=1.1,
                           stop=["### Instruction:", "### Input:"])
            candidate = out["choices"][0]["text"].strip()
            if candidate and guard_ok(text, candidate):
                return {"text": candidate,
                        "guard": "passed" if temp == 0.0 else f"passed_retry_t{temp}"}
        return {"text": text, "guard": "failed_passthrough"}


ENGINE = None


class Handler(BaseHTTPRequestHandler):
    def log_message(self, fmt, *args):  # quiet default access log
        pass

    def do_POST(self):
        if self.path.rstrip("/") not in ("/v1/chat/completions", "/chat/completions"):
            self.send_error(404)
            return
        try:
            body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
            user_texts = [m.get("content", "") for m in body.get("messages", [])
                          if m.get("role") == "user"]
            source = user_texts[-1] if user_texts else ""
        except Exception:  # noqa: BLE001 - malformed request
            self.send_error(400)
            return

        t0 = time.time()
        result = ENGINE.rewrite(source)
        print(f"[{time.strftime('%H:%M:%S')}] {len(source)} chars -> "
              f"{len(result['text'])} chars · guard={result['guard']} · "
              f"{time.time()-t0:.1f}s", flush=True)

        response = {
            "id": "chatcmpl-" + uuid.uuid4().hex[:12],
            "object": "chat.completion",
            "created": int(time.time()),
            "model": "claudish-style-adapter-v3",
            "choices": [{"index": 0, "finish_reason": "stop",
                         "message": {"role": "assistant", "content": result["text"]}}],
            "claudish_guard": result["guard"],
        }
        payload = json.dumps(response).encode()
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(payload)))
        self.end_headers()
        self.wfile.write(payload)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--port", type=int, default=8017)
    args = parser.parse_args()
    global ENGINE
    print("loading model ...", flush=True)
    ENGINE = Engine()
    print(f"claudish adapter serving on http://localhost:{args.port}/v1/chat/completions", flush=True)
    ThreadingHTTPServer(("127.0.0.1", args.port), Handler).serve_forever()


if __name__ == "__main__":
    main()
