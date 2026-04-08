from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path

from leopardi.ops import (
    ArtifactPointer,
    RunHeartbeat,
    RunManifest,
    RunSummary,
    append_event,
    ensure_run_layout,
    write_heartbeat,
    write_manifest,
    write_summary,
)
from leopardi.optimization.artifacts import OptimizationArtifactCard, artifact_card_dict, build_artifact_card
from leopardi.optimization.config import OptimizationStageConfig, OptimizationVariantConfig


@dataclass(slots=True)
class OptimizationVariantPlan:
    variant_id: str
    backend_family: str
    artifact_dir: str
    artifact_card_path: str
    command_path: str
    report_stub_path: str
    persistent_artifact_uri: str
    runtime_targets: tuple[str, ...]
    commands: tuple[str, ...]


def _shell_quote(value: str) -> str:
    return "'" + value.replace("'", "'\"'\"'") + "'"


def _default_local_checkpoint_dir(base_checkpoint_uri: str) -> str:
    if base_checkpoint_uri.startswith("hf://"):
        return "/mnt/leopardi/checkpoints/base"
    return base_checkpoint_uri


def _python_loader_target(base_checkpoint_uri: str) -> str:
    default_dir = _default_local_checkpoint_dir(base_checkpoint_uri)
    return f"os.environ.get('LEOPARDI_BASE_CHECKPOINT_DIR', {_shell_quote(default_dir)})"


def _backend_family(method: str) -> str:
    if method.startswith("llmcompressor"):
        return "llmcompressor"
    if method.startswith("torchao"):
        return "torchao"
    if method == "runtime_kv_quant":
        return "runtime_only"
    if method == "reference_export":
        return "reference"
    return "custom"


def _torchao_commands(
    *,
    variant: OptimizationVariantConfig,
    base_checkpoint_uri: str,
    artifact_dir: Path,
) -> tuple[str, ...]:
    snippet = f"""python - <<'PY'
import os
import torch
from transformers import AutoModelForCausalLM, AutoProcessor
from torchao.quantization import Int4WeightOnlyConfig, Float8DynamicActivationFloat8WeightConfig, PerRow, quantize_

base_checkpoint = {_python_loader_target(base_checkpoint_uri)}
model = AutoModelForCausalLM.from_pretrained(base_checkpoint, torch_dtype=torch.bfloat16, device_map='cuda')
try:
    processor = AutoProcessor.from_pretrained(base_checkpoint)
except Exception:
    processor = None

if {_shell_quote(variant.variant_id)} == 'torchao_int4_weight_only':
    quantize_(model, Int4WeightOnlyConfig(group_size=32, int4_packing_format='tile_packed_to_4d', int4_choose_qparams_algorithm='hqq'))
else:
    quantize_(model, Float8DynamicActivationFloat8WeightConfig(granularity=PerRow()))

model.save_pretrained({_shell_quote(str(artifact_dir))}, safe_serialization=False)
if processor is not None:
    processor.save_pretrained({_shell_quote(str(artifact_dir))})
PY"""
    preamble = (
        "# Before running this plan, materialize the promoted base checkpoint on local disk.\n"
        f"export LEOPARDI_BASE_CHECKPOINT_DIR={_shell_quote(_default_local_checkpoint_dir(base_checkpoint_uri))}"
    )
    serve_hint = (
        f"VLLM_DISABLE_COMPILE_CACHE=1 vllm serve {_shell_quote(str(artifact_dir))} "
        f"--structured-outputs-config.backend {variant.structured_output_backend}"
    )
    return (preamble, snippet, serve_hint)


def _llmcompressor_commands(
    *,
    variant: OptimizationVariantConfig,
    base_checkpoint_uri: str,
    artifact_dir: Path,
    calibration_bundle_id: str,
    calibration_samples: int,
) -> tuple[str, ...]:
    dataset_hint = (
        "os.environ.get('LEOPARDI_CALIBRATION_DATASET', "
        f"{_shell_quote('/mnt/leopardi/calibration/' + calibration_bundle_id)})"
    )
    if variant.method == "llmcompressor_fp8":
        recipe = "scheme='FP8_BLOCK'"
        oneshot_args = ""
    else:
        recipe = "scheme='W4A16', algorithm='AWQ'"
        oneshot_args = (
            f", dataset={dataset_hint}, num_calibration_samples={calibration_samples}"
        )

    snippet = f"""python - <<'PY'
import os

from transformers import AutoModelForCausalLM, AutoProcessor
from llmcompressor import oneshot
from llmcompressor.modifiers.quantization import QuantizationModifier

base_checkpoint = {_python_loader_target(base_checkpoint_uri)}
model = AutoModelForCausalLM.from_pretrained(base_checkpoint, dtype='auto')
try:
    processor = AutoProcessor.from_pretrained(base_checkpoint)
except Exception:
    processor = None

recipe = QuantizationModifier(
    targets='Linear',
    {recipe},
    ignore=['lm_head'],
)
oneshot(model=model, recipe=recipe{oneshot_args})
model.save_pretrained({_shell_quote(str(artifact_dir))}, save_compressed=True)
if processor is not None:
    processor.save_pretrained({_shell_quote(str(artifact_dir))})
PY"""
    preamble = (
        "# Before running this plan, materialize the promoted base checkpoint on local disk.\n"
        f"export LEOPARDI_BASE_CHECKPOINT_DIR={_shell_quote(_default_local_checkpoint_dir(base_checkpoint_uri))}\n"
        "# Only required for calibration-based variants.\n"
        f"export LEOPARDI_CALIBRATION_DATASET={_shell_quote('/mnt/leopardi/calibration/' + calibration_bundle_id)}"
    )
    serve_hint = f"vllm serve {_shell_quote(str(artifact_dir))}"
    return (preamble, snippet, serve_hint)


def _runtime_kv_commands(
    *,
    variant: OptimizationVariantConfig,
    base_checkpoint_uri: str,
) -> tuple[str, ...]:
    preamble = (
        "# Point this to the already-materialized optimized checkpoint you want to benchmark.\n"
        f"export LEOPARDI_BASE_CHECKPOINT_DIR={_shell_quote(_default_local_checkpoint_dir(base_checkpoint_uri))}"
    )
    if "sglang" in variant.runtime_targets:
        return (
            preamble,
            "python -m sglang.launch_server --model-path \"$LEOPARDI_BASE_CHECKPOINT_DIR\" "
            "--kv-cache-dtype fp8 --grammar-backend xgrammar",
        )
    return (
        preamble,
        "vllm serve \"$LEOPARDI_BASE_CHECKPOINT_DIR\" "
        "--kv-cache-dtype fp8 --structured-outputs-config.backend xgrammar",
    )


def _reference_commands(*, base_checkpoint_uri: str, artifact_dir: Path) -> tuple[str, ...]:
    return (
        (
            "# Export the promoted bf16 checkpoint into a pinned serving directory.\n"
            f"# Source checkpoint: {base_checkpoint_uri}\n"
            f"# Target artifact dir: {artifact_dir}"
        ),
        f"vllm serve {_shell_quote(str(artifact_dir))}",
    )


def build_variant_commands(
    *,
    stage: OptimizationStageConfig,
    variant: OptimizationVariantConfig,
    base_checkpoint_uri: str,
    artifact_dir: str | Path,
) -> tuple[str, ...]:
    artifact_path = Path(artifact_dir)
    calibration_samples = variant.calibration_samples or stage.calibration.max_samples
    if variant.method == "reference_export":
        return _reference_commands(base_checkpoint_uri=base_checkpoint_uri, artifact_dir=artifact_path)
    if variant.method.startswith("torchao"):
        return _torchao_commands(
            variant=variant,
            base_checkpoint_uri=base_checkpoint_uri,
            artifact_dir=artifact_path,
        )
    if variant.method.startswith("llmcompressor"):
        return _llmcompressor_commands(
            variant=variant,
            base_checkpoint_uri=base_checkpoint_uri,
            artifact_dir=artifact_path,
            calibration_bundle_id=stage.calibration.bundle_id,
            calibration_samples=calibration_samples,
        )
    if variant.method == "runtime_kv_quant":
        return _runtime_kv_commands(variant=variant, base_checkpoint_uri=base_checkpoint_uri)
    raise ValueError(f"Unsupported optimization method: {variant.method}")


def build_variant_runtime_plan(
    *,
    experiment_id: str,
    stage: OptimizationStageConfig,
    variant: OptimizationVariantConfig,
    artifacts_root: str | Path,
    persistent_root: str,
    base_checkpoint_uri: str,
) -> tuple[OptimizationVariantPlan, OptimizationArtifactCard]:
    artifacts_root = Path(artifacts_root)
    variant_dir = artifacts_root / "optimization" / variant.variant_id
    card_path = variant_dir / "artifact-card.json"
    command_path = variant_dir / "commands.sh"
    report_stub_path = variant_dir / "report.stub.json"
    persistent_uri = f"{persistent_root.rstrip('/')}/{experiment_id}/{stage.stage}/{variant.variant_id}"
    card = build_artifact_card(
        experiment_id=experiment_id,
        stage=stage,
        variant=variant,
        artifact_dir=variant_dir,
        persistent_artifact_uri=persistent_uri,
        base_checkpoint_uri=base_checkpoint_uri,
    )
    commands = build_variant_commands(
        stage=stage,
        variant=variant,
        base_checkpoint_uri=base_checkpoint_uri,
        artifact_dir=variant_dir,
    )
    plan = OptimizationVariantPlan(
        variant_id=variant.variant_id,
        backend_family=_backend_family(variant.method),
        artifact_dir=str(variant_dir),
        artifact_card_path=str(card_path),
        command_path=str(command_path),
        report_stub_path=str(report_stub_path),
        persistent_artifact_uri=persistent_uri,
        runtime_targets=variant.runtime_targets,
        commands=commands,
    )
    return plan, card


def _write_json(path: Path, payload: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def materialize_optimization_stage(
    *,
    experiment_id: str,
    stage: OptimizationStageConfig,
    base_checkpoint_uri: str,
    stage_config_path: str | None = None,
    runtime_config_path: str | None = None,
    root: str | Path = "runs",
) -> dict[str, object]:
    layout = ensure_run_layout(experiment_id, root=root)
    manifest = RunManifest(
        experiment_id=experiment_id,
        phase="optimization",
        stage=stage.stage,
        track=stage.track,
        hardware_tag=stage.runtime.hardware_tag,
        config_paths=[
            stage_config_path or f"generated::optimization::{stage.stage}",
            runtime_config_path or "generated::runtime::optimization",
        ],
        data_bundle_ids=[stage.calibration.bundle_id] if any(v.requires_calibration for v in stage.variants) else [],
        protocol_version="release_gate_v1",
        local_run_root=str(layout.experiment_root),
        persistent_targets={
            "checkpoints": "hf://leopardi-ocr-checkpoints",
            "reports": "hf://leopardi-ocr-reports",
            "metadata": "hf://leopardi-ocr-metadata",
        },
    )
    write_manifest(manifest, layout=layout)
    write_heartbeat(
        RunHeartbeat(
            experiment_id=experiment_id,
            phase="optimization",
            stage=stage.stage,
            state="draft",
            current_step=0,
        ),
        layout=layout,
    )

    plans: list[OptimizationVariantPlan] = []
    cards: list[OptimizationArtifactCard] = []
    for variant in stage.variants:
        plan, card = build_variant_runtime_plan(
            experiment_id=experiment_id,
            stage=stage,
            variant=variant,
            artifacts_root=layout.artifacts_dir,
            persistent_root="hf://leopardi-ocr-checkpoints",
            base_checkpoint_uri=base_checkpoint_uri,
        )
        plans.append(plan)
        cards.append(card)
        variant_dir = Path(plan.artifact_dir)
        variant_dir.mkdir(parents=True, exist_ok=True)
        _write_json(Path(plan.artifact_card_path), artifact_card_dict(card))
        Path(plan.command_path).write_text("\n\n".join(plan.commands) + "\n", encoding="utf-8")
        _write_json(
            Path(plan.report_stub_path),
            {
                "experiment_id": experiment_id,
                "stage": stage.stage,
                "variant_id": variant.variant_id,
                "status": "pending_measurement",
                "required_protocols": ["public_frontier_v1", "release_gate_v1"],
            },
        )
        append_event(
            layout=layout,
            event_type="optimization_variant_planned",
            phase="optimization",
            stage=stage.stage,
            payload={
                "variant_id": variant.variant_id,
                "backend_family": plan.backend_family,
                "persistent_artifact_uri": plan.persistent_artifact_uri,
            },
        )

    write_summary(
        RunSummary(
            experiment_id=experiment_id,
            phase="optimization",
            stage=stage.stage,
            outcome="completed",
            key_metrics={},
            artifacts=[
                ArtifactPointer(
                    artifact_kind="release_card",
                    uri=f"local://{plan.artifact_card_path}",
                    local_path=plan.artifact_card_path,
                    persistence_status="local_only",
                )
                for plan in plans
            ],
            notes=[
                "Optimization planning materialized successfully.",
                "Command plans are ready for operator execution on a GPU-backed machine.",
            ],
        ),
        layout=layout,
    )
    return {
        "layout": layout.as_dict(),
        "plans": [asdict(plan) for plan in plans],
        "cards": [artifact_card_dict(card) for card in cards],
    }
