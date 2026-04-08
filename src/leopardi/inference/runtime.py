from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path

from leopardi.inference.artifacts import (
    build_inference_artifact_card,
    inference_artifact_card_dict,
)
from leopardi.inference.config import InferenceStageConfig
from leopardi.inference.routing import mode_summary
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


@dataclass(slots=True)
class InferenceLaunchPlan:
    runtime_family: str
    mode_profiles: tuple[str, ...]
    launch_command: str
    healthcheck_command: str
    sample_request_path: str
    report_stub_path: str
    runtime_plan_path: str
    runtime_report_uri: str


def _shell_quote(value: str) -> str:
    return "'" + value.replace("'", "'\"'\"'") + "'"


def _default_model_dir(artifact_uri: str) -> str:
    if artifact_uri.startswith("hf://"):
        return "/mnt/leopardi/model"
    return artifact_uri


def _write_json(path: Path, payload: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _vllm_launch_command(stage: InferenceStageConfig) -> str:
    return (
        "VLLM_DISABLE_COMPILE_CACHE=1 "
        f"vllm serve \"${{{stage.runtime.model_dir_env}}}\" "
        f"--host {stage.runtime.host} "
        f"--port {stage.runtime.vllm_port} "
        f"--gpu-memory-utilization {stage.runtime.gpu_memory_utilization:.2f} "
        f"--max-model-len {stage.runtime.max_model_len} "
        f"--max-num-seqs {stage.runtime.max_num_seqs} "
        f"--tensor-parallel-size {stage.runtime.tensor_parallel_size} "
        f"--structured-outputs-config.backend {stage.structured_backend_default}"
    )


def _sglang_launch_command(stage: InferenceStageConfig) -> str:
    prefix = "--enable-prefix-caching" if stage.runtime.enable_prefix_caching else ""
    return (
        "python -m sglang.launch_server "
        f"--model-path \"${{{stage.runtime.model_dir_env}}}\" "
        f"--host {stage.runtime.host} "
        f"--port {stage.runtime.sglang_port} "
        f"--context-len {stage.runtime.max_model_len} "
        f"--mem-fraction-static {stage.runtime.mem_fraction_static:.2f} "
        f"--grammar-backend {stage.structured_backend_default} "
        f"{prefix}"
    ).strip()


def build_launch_plan(
    *,
    experiment_id: str,
    stage: InferenceStageConfig,
    runtime_family: str,
    artifacts_root: str | Path,
    persistent_report_root: str,
) -> InferenceLaunchPlan:
    artifacts_root = Path(artifacts_root)
    runtime_dir = artifacts_root / "inference" / runtime_family
    sample_request_path = runtime_dir / "sample-requests.json"
    report_stub_path = runtime_dir / "report.stub.json"
    runtime_plan_path = runtime_dir / "runtime-plan.json"
    runtime_report_uri = f"{persistent_report_root.rstrip('/')}/{experiment_id}/{stage.stage}/{runtime_family}"
    if runtime_family == "vllm":
        launch_command = _vllm_launch_command(stage)
        healthcheck_command = (
            f"curl -fsS http://{stage.runtime.host}:{stage.runtime.vllm_port}/health || exit 1"
        )
    elif runtime_family == "sglang":
        launch_command = _sglang_launch_command(stage)
        healthcheck_command = (
            f"curl -fsS http://{stage.runtime.host}:{stage.runtime.sglang_port}/health || exit 1"
        )
    else:
        raise ValueError(f"Unsupported inference runtime: {runtime_family}")
    return InferenceLaunchPlan(
        runtime_family=runtime_family,
        mode_profiles=tuple(mode.name for mode in stage.modes),
        launch_command=launch_command,
        healthcheck_command=healthcheck_command,
        sample_request_path=str(sample_request_path),
        report_stub_path=str(report_stub_path),
        runtime_plan_path=str(runtime_plan_path),
        runtime_report_uri=runtime_report_uri,
    )


def materialize_inference_stage(
    *,
    experiment_id: str,
    stage: InferenceStageConfig,
    stage_config_path: str | None = None,
    runtime_config_path: str | None = None,
    root: str | Path = "runs",
) -> dict[str, object]:
    layout = ensure_run_layout(experiment_id, root=root)
    write_manifest(
        RunManifest(
            experiment_id=experiment_id,
            phase="inference",
            stage=stage.stage,
            track=stage.track,
            hardware_tag=stage.runtime.hardware_tag,
            config_paths=[
                stage_config_path or f"generated::inference::{stage.stage}",
                runtime_config_path or "generated::runtime::inference",
            ],
            data_bundle_ids=[],
            protocol_version="public_frontier_v1",
            local_run_root=str(layout.experiment_root),
            persistent_targets={
                "reports": "hf://leopardi-ocr-reports",
                "metadata": "hf://leopardi-ocr-metadata",
            },
        ),
        layout=layout,
    )
    write_heartbeat(
        RunHeartbeat(
            experiment_id=experiment_id,
            phase="inference",
            stage=stage.stage,
            state="draft",
            current_step=0,
        ),
        layout=layout,
    )

    runtimes = (stage.runtime_family, stage.fallback_runtime_family)
    plans = [
        build_launch_plan(
            experiment_id=experiment_id,
            stage=stage,
            runtime_family=runtime_family,
            artifacts_root=layout.artifacts_dir,
            persistent_report_root="hf://leopardi-ocr-reports",
        )
        for runtime_family in dict.fromkeys(runtimes)
    ]
    mode_summaries = tuple(mode_summary(mode) for mode in stage.modes)
    artifact_card = build_inference_artifact_card(
        experiment_id=experiment_id,
        stage=stage,
        primary_report_uri=plans[0].runtime_report_uri,
        metadata_uri=f"hf://leopardi-ocr-metadata/{experiment_id}/{stage.stage}",
        mode_summaries=mode_summaries,
    )

    card_path = layout.artifacts_dir / "inference" / "artifact-card.json"
    _write_json(card_path, inference_artifact_card_dict(artifact_card))
    for plan in plans:
        runtime_dir = Path(plan.runtime_plan_path).parent
        runtime_dir.mkdir(parents=True, exist_ok=True)
        _write_json(
            Path(plan.runtime_plan_path),
            {
                "runtime_family": plan.runtime_family,
                "mode_profiles": list(plan.mode_profiles),
                "launch_command": plan.launch_command,
                "healthcheck_command": plan.healthcheck_command,
                "model_dir_env": stage.runtime.model_dir_env,
                "default_model_dir": _default_model_dir(stage.artifact_uri),
            },
        )
        _write_json(
            Path(plan.sample_request_path),
            {
                "host": stage.runtime.host,
                "runtime_family": plan.runtime_family,
                "requests": [
                    {
                        "mode": mode.name,
                        "image_path_env": "LEOPARDI_PAGE_IMAGE",
                        "max_output_tokens": mode.max_output_tokens,
                        "grammar_mode": mode.grammar_mode,
                        "structured_output_backend": mode.structured_output_backend,
                        "repair_budget": mode.repair_budget,
                    }
                    for mode in stage.modes
                ],
            },
        )
        _write_json(
            Path(plan.report_stub_path),
            {
                "experiment_id": experiment_id,
                "stage": stage.stage,
                "runtime_family": plan.runtime_family,
                "status": "pending_runtime_measurement",
                "required_protocols": ["public_frontier_v1", "release_gate_v1"],
            },
        )
        append_event(
            layout=layout,
            event_type="inference_runtime_planned",
            phase="inference",
            stage=stage.stage,
            payload={
                "runtime_family": plan.runtime_family,
                "runtime_report_uri": plan.runtime_report_uri,
            },
        )

    write_summary(
        RunSummary(
            experiment_id=experiment_id,
            phase="inference",
            stage=stage.stage,
            outcome="completed",
            key_metrics={},
            artifacts=[
                ArtifactPointer(
                    artifact_kind="release_card",
                    uri=f"local://{card_path}",
                    local_path=str(card_path),
                    persistence_status="local_only",
                )
            ],
            notes=[
                "Inference launch plans materialized successfully.",
                "Use the generated runtime plans to launch production and evaluation servers on the rented GPU machine.",
            ],
        ),
        layout=layout,
    )
    return {
        "layout": layout.as_dict(),
        "artifact_card": inference_artifact_card_dict(artifact_card),
        "plans": [asdict(plan) for plan in plans],
    }
