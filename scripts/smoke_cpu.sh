#!/usr/bin/env bash
set -euo pipefail

PYTHON_BIN="python3"
RUFF_BIN="ruff"

if [ -x ".venv/bin/python" ]; then
  PYTHON_BIN=".venv/bin/python"
fi

if [ -x ".venv/bin/ruff" ]; then
  RUFF_BIN=".venv/bin/ruff"
fi

"$PYTHON_BIN" -m pytest -q
"$RUFF_BIN" check src tests ops docs configs experiments evaluation data_pipeline pretraining finetune optimization scripts
"$PYTHON_BIN" -m leopardi.cli doctor
"$PYTHON_BIN" -m leopardi.cli schema-example >/dev/null
"$PYTHON_BIN" -m leopardi.cli model-summary configs/model/leopardi_s0.yaml >/dev/null
"$PYTHON_BIN" -m leopardi.cli pretrain-summary configs/pretraining/s0_p2_multimodal_core.yaml configs/runtime/train_rtx5090.yaml >/dev/null
"$PYTHON_BIN" -m leopardi.cli finetune-summary configs/finetune/s0_f0_sft.yaml configs/runtime/finetune_rtx5090.yaml >/dev/null
"$PYTHON_BIN" -m leopardi.cli optimization-summary configs/optimization/s0_o2_vllm_compressed.yaml configs/runtime/optimization_rtx5090.yaml >/dev/null
"$PYTHON_BIN" -m leopardi.cli optimization-plan configs/optimization/s0_o2_vllm_compressed.yaml configs/runtime/optimization_rtx5090.yaml hf://leopardi-ocr-checkpoints/leo-s0-f3-candidate >/dev/null
"$PYTHON_BIN" -m leopardi.cli smoke-train-step configs/model/leopardi_s0.yaml configs/pretraining/s0_p2_multimodal_core.yaml configs/runtime/train_rtx5090.yaml >/dev/null
"$PYTHON_BIN" -m leopardi.cli smoke-finetune-step configs/model/leopardi_s0.yaml configs/finetune/s0_f0_sft.yaml configs/runtime/finetune_rtx5090.yaml >/dev/null
"$PYTHON_BIN" -m leopardi.cli run-layout leo-s0-smoke-20260408-001 >/dev/null
"$PYTHON_BIN" -m leopardi.cli ops-examples >/dev/null
"$PYTHON_BIN" -m leopardi.cli optimization-rank-example >/dev/null
"$PYTHON_BIN" -m leopardi.cli optimization-materialize leo-s0-o2-smoke-20260408-001 configs/optimization/s0_o2_vllm_compressed.yaml configs/runtime/optimization_rtx5090.yaml hf://leopardi-ocr-checkpoints/leo-s0-f3-candidate --root tmp/optimization-smoke >/dev/null
"$PYTHON_BIN" -m leopardi.cli materialize-run-example --root tmp/ops-smoke >/dev/null
rm -rf tmp/optimization-smoke
rm -rf tmp/ops-smoke
