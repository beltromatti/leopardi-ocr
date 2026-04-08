#!/usr/bin/env bash
set -euo pipefail

python3 -m venv .venv
.venv/bin/python -m pip install --upgrade pip
.venv/bin/pip install -e ".[dev,train]"

echo "Installed control-plane and training dependencies."
echo "Use ./scripts/bootstrap_rtx5090.sh on the rented GPU machine."
