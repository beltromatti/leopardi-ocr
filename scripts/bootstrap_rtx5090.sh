#!/usr/bin/env bash
set -euo pipefail

python3 -m venv .venv
.venv/bin/python -m pip install --upgrade pip uv
.venv/bin/pip install -e ".[dev,train]"

# GPU-only runtime and post-training stack.
.venv/bin/uv pip install vllm --torch-backend=auto
.venv/bin/pip install llmcompressor
.venv/bin/pip install torchao
.venv/bin/uv pip install "sglang[all]"

.venv/bin/python - <<'PY'
mods = ["torch", "transformers", "datasets", "huggingface_hub", "torchao", "llmcompressor", "vllm", "sglang"]
for mod in mods:
    module = __import__(mod)
    print(f"{mod}: {getattr(module, '__version__', 'unknown')}")
PY
