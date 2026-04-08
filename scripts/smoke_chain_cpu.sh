#!/usr/bin/env bash
set -euo pipefail

PYTHON_BIN="python3"

if [ -x ".venv/bin/python" ]; then
  PYTHON_BIN=".venv/bin/python"
fi

ROOT="tmp/chain-smoke"
rm -rf "$ROOT"

"$PYTHON_BIN" -m leopardi.cli data-pipeline-materialize \
  leo-s0-data-chain-smoke \
  configs/data/s0_full_frontier_build.yaml \
  configs/runtime/data_build_rtx5090.yaml \
  --root "$ROOT" >/dev/null

"$PYTHON_BIN" -m leopardi.cli pretrain-materialize \
  leo-s0-p2-chain-smoke \
  configs/pretraining/s0_p2_multimodal_core.yaml \
  configs/runtime/train_rtx5090.yaml \
  configs/model/leopardi_s0.yaml \
  --root "$ROOT" >/dev/null

"$PYTHON_BIN" -m leopardi.cli finetune-materialize \
  leo-s0-f3-chain-smoke \
  configs/finetune/s0_f3_rlvr.yaml \
  configs/runtime/finetune_rtx5090.yaml \
  configs/model/leopardi_s0.yaml \
  --root "$ROOT" >/dev/null

"$PYTHON_BIN" -m leopardi.cli optimization-materialize \
  leo-s0-o2-chain-smoke \
  configs/optimization/s0_o2_vllm_compressed.yaml \
  configs/runtime/optimization_rtx5090.yaml \
  hf://leopardi-ocr-checkpoints/leo-s0-f3-chain-smoke/f3_rlvr \
  --root "$ROOT" >/dev/null

"$PYTHON_BIN" -m leopardi.cli inference-materialize \
  leo-s0-i1-chain-smoke \
  configs/inference/s0_i1_vllm_adaptive.yaml \
  configs/runtime/inference_rtx5090.yaml \
  --root "$ROOT" >/dev/null

"$PYTHON_BIN" -m leopardi.cli evaluation-materialize \
  leo-s0-eval-chain-smoke \
  configs/eval/public_frontier.yaml \
  configs/runtime/eval_rtx5090.yaml \
  --root "$ROOT" >/dev/null

"$PYTHON_BIN" - <<'PY'
import json
from pathlib import Path

root = Path("tmp/chain-smoke")

def read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))

data_plan = read_json(root / "leo-s0-data-chain-smoke" / "artifacts" / "data_pipeline" / "s0_full_frontier_build" / "data-build-plan.json")
pretrain_manifest = read_json(root / "leo-s0-p2-chain-smoke" / "manifest.json")
finetune_manifest = read_json(root / "leo-s0-f3-chain-smoke" / "manifest.json")
optimization_manifest = read_json(root / "leo-s0-o2-chain-smoke" / "manifest.json")
inference_card = read_json(root / "leo-s0-i1-chain-smoke" / "artifacts" / "inference" / "artifact-card.json")
evaluation_manifest = read_json(root / "leo-s0-eval-chain-smoke" / "manifest.json")

built_bundles = set(data_plan["bundle_ids"])
assert set(pretrain_manifest["data_bundle_ids"]).issubset(built_bundles)
assert set(finetune_manifest["data_bundle_ids"]).issubset(built_bundles)
assert optimization_manifest["data_bundle_ids"] == ["o_calibration_docmix_v1"]
assert inference_card["artifact_variant_id"] == "llmcompressor_fp8_dynamic"
assert evaluation_manifest["data_bundle_ids"] == ["public_frontier_v1"]

print("Full chain control-plane check passed.")
PY

rm -rf "$ROOT"
echo "Full chain smoke passed!"
