from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path

import torch
from torch import nn

from leopardi.finetune.config import FinetuneStageConfig
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
class FinetuneExecutionPlan:
    stage: str
    track: str
    data_bundle_ids: tuple[str, ...]
    effective_batch_size: int
    train_command: str
    plan_path: str
    report_stub_path: str
    checkpoint_stub_dir: str
    checkpoint_uri: str


def finetune_effective_batch_size(stage_config: FinetuneStageConfig) -> int:
    return (
        stage_config.runtime.micro_batch_size * stage_config.runtime.gradient_accumulation_steps
    )


def _write_json(path: Path, payload: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _named_trainable_parameters(model: nn.Module) -> list[tuple[str, nn.Parameter]]:
    return [(name, parameter) for name, parameter in model.named_parameters() if parameter.requires_grad]


def _module_scale_for_name(name: str, stage_config: FinetuneStageConfig) -> float:
    scales = stage_config.module_lr
    if name.startswith("visual_tokenizer"):
        return scales.visual_tokenizer
    if name.startswith("layout_side_encoder"):
        return scales.layout_side_encoder
    if name.startswith("latent_bottleneck"):
        return scales.latent_bottleneck
    if name.startswith("planner"):
        return scales.planner
    if name.startswith("writer"):
        return scales.writer
    return scales.auxiliary_heads


def _uses_no_decay(name: str, stage_config: FinetuneStageConfig) -> bool:
    if not stage_config.module_lr.no_decay_on_norms_and_bias:
        return False
    return name.endswith(".bias") or ".norm." in name or "layernorm" in name.lower()


def optimizer_group_summary(model: nn.Module, stage_config: FinetuneStageConfig) -> list[dict[str, object]]:
    buckets: dict[tuple[float, float], dict[str, object]] = {}
    for name, parameter in _named_trainable_parameters(model):
        if stage_config.adapter.mode != "full" and not any(
            name.startswith(target) for target in stage_config.adapter.target_modules
        ):
            continue
        lr_scale = _module_scale_for_name(name, stage_config)
        weight_decay = 0.0 if _uses_no_decay(name, stage_config) else stage_config.optimizer.weight_decay
        key = (lr_scale, weight_decay)
        if key not in buckets:
            buckets[key] = {
                "lr_scale": lr_scale,
                "weight_decay": weight_decay,
                "parameter_count": 0,
                "sample_names": [],
            }
        bucket = buckets[key]
        bucket["parameter_count"] += parameter.numel()
        if len(bucket["sample_names"]) < 5:
            bucket["sample_names"].append(name)
    return sorted(
        buckets.values(),
        key=lambda item: (float(item["lr_scale"]), float(item["weight_decay"])),
    )


def build_finetune_optimizer(
    model: nn.Module,
    stage_config: FinetuneStageConfig,
) -> torch.optim.Optimizer:
    optimizer_cfg = stage_config.optimizer
    if optimizer_cfg.name.lower() != "adamw":
        raise ValueError(f"Unsupported finetune optimizer: {optimizer_cfg.name}")

    param_groups: dict[tuple[float, float], dict[str, object]] = {}
    for name, parameter in _named_trainable_parameters(model):
        if stage_config.adapter.mode != "full" and not any(
            name.startswith(target) for target in stage_config.adapter.target_modules
        ):
            continue
        lr_scale = _module_scale_for_name(name, stage_config)
        weight_decay = 0.0 if _uses_no_decay(name, stage_config) else optimizer_cfg.weight_decay
        key = (lr_scale, weight_decay)
        if key not in param_groups:
            param_groups[key] = {
                "params": [],
                "lr": optimizer_cfg.lr * lr_scale,
                "weight_decay": weight_decay,
            }
        param_groups[key]["params"].append(parameter)

    return torch.optim.AdamW(
        list(param_groups.values()),
        lr=optimizer_cfg.lr,
        betas=optimizer_cfg.betas,
        eps=optimizer_cfg.eps,
    )


def apply_finetune_runtime_policy(model: nn.Module, stage_config: FinetuneStageConfig) -> nn.Module:
    model.train()
    if stage_config.adapter.mode != "full":
        targets = stage_config.adapter.target_modules
        for name, parameter in model.named_parameters():
            parameter.requires_grad = any(name.startswith(target) for target in targets)
    if stage_config.runtime.compile_model:
        model = torch.compile(model)
    return model


def build_finetune_execution_plan(
    *,
    experiment_id: str,
    stage: FinetuneStageConfig,
    model_config_path: str,
    stage_config_path: str,
    runtime_config_path: str,
    root: str | Path = "runs",
) -> FinetuneExecutionPlan:
    layout = ensure_run_layout(experiment_id, root=root)
    plan_dir = layout.artifacts_dir / "finetune" / stage.stage
    checkpoint_uri = f"hf://leopardi-ocr-checkpoints/{experiment_id}/{stage.stage}"
    train_command = (
        "python -m leopardi.cli smoke-finetune-step "
        f"{model_config_path} {stage_config_path} {runtime_config_path}"
    )
    return FinetuneExecutionPlan(
        stage=stage.stage,
        track=stage.track,
        data_bundle_ids=stage.data_bundle_ids,
        effective_batch_size=finetune_effective_batch_size(stage),
        train_command=train_command,
        plan_path=str(plan_dir / "finetune-plan.json"),
        report_stub_path=str(plan_dir / "report.stub.json"),
        checkpoint_stub_dir=str(plan_dir / "checkpoints"),
        checkpoint_uri=checkpoint_uri,
    )


def materialize_finetune_stage(
    *,
    experiment_id: str,
    stage: FinetuneStageConfig,
    model_config_path: str = "configs/model/leopardi_s0.yaml",
    stage_config_path: str | None = None,
    runtime_config_path: str | None = None,
    root: str | Path = "runs",
) -> dict[str, object]:
    layout = ensure_run_layout(experiment_id, root=root)
    plan = build_finetune_execution_plan(
        experiment_id=experiment_id,
        stage=stage,
        model_config_path=model_config_path,
        stage_config_path=stage_config_path or f"generated::finetune::{stage.stage}",
        runtime_config_path=runtime_config_path or "generated::runtime::finetune",
        root=root,
    )
    write_manifest(
        RunManifest(
            experiment_id=experiment_id,
            phase="finetune",
            stage=stage.stage,
            track=stage.track,
            hardware_tag=stage.runtime.hardware_tag,
            config_paths=[
                model_config_path,
                stage_config_path or f"generated::finetune::{stage.stage}",
                runtime_config_path or "generated::runtime::finetune",
            ],
            data_bundle_ids=list(stage.data_bundle_ids),
            protocol_version="internal_holdout_v1",
            local_run_root=str(layout.experiment_root),
            persistent_targets={
                "checkpoints": "hf://leopardi-ocr-checkpoints",
                "reports": "hf://leopardi-ocr-reports",
                "metadata": "hf://leopardi-ocr-metadata",
            },
        ),
        layout=layout,
    )
    write_heartbeat(
        RunHeartbeat(
            experiment_id=experiment_id,
            phase="finetune",
            stage=stage.stage,
            state="draft",
            current_step=0,
        ),
        layout=layout,
    )
    _write_json(
        Path(plan.plan_path),
        {
            "stage": stage.stage,
            "track": stage.track,
            "visual_mode": stage.visual_mode,
            "adapter": asdict(stage.adapter),
            "effective_batch_size": plan.effective_batch_size,
            "optimizer": asdict(stage.optimizer),
            "scheduler": asdict(stage.scheduler),
            "sampling": asdict(stage.sampling),
            "data_bundle_ids": list(stage.data_bundle_ids),
            "module_lr": asdict(stage.module_lr),
            "verifier": asdict(stage.verifier),
            "loss_weights": asdict(stage.loss_weights),
            "reward_weights": asdict(stage.reward_weights),
            "train_command": plan.train_command,
            "checkpoint_uri": plan.checkpoint_uri,
        },
    )
    _write_json(
        Path(plan.report_stub_path),
        {
            "experiment_id": experiment_id,
            "stage": stage.stage,
            "status": "pending_finetune",
            "quality_focus": [
                "markdown_exactness",
                "formula_exactness",
                "table_exactness",
                "repair_efficiency",
            ],
        },
    )
    append_event(
        layout=layout,
        event_type="finetune_plan_materialized",
        phase="finetune",
        stage=stage.stage,
        payload={
            "effective_batch_size": plan.effective_batch_size,
            "checkpoint_uri": plan.checkpoint_uri,
            "adapter_mode": stage.adapter.mode,
        },
    )
    write_summary(
        RunSummary(
            experiment_id=experiment_id,
            phase="finetune",
            stage=stage.stage,
            outcome="completed",
            key_metrics={},
            artifacts=[
                ArtifactPointer(
                    artifact_kind="runtime_plan",
                    uri=f"local://{plan.plan_path}",
                    local_path=plan.plan_path,
                    persistence_status="local_only",
                ),
                ArtifactPointer(
                    artifact_kind="checkpoint",
                    uri=plan.checkpoint_uri,
                    local_path=plan.checkpoint_stub_dir,
                    persistence_status="queued",
                ),
                ArtifactPointer(
                    artifact_kind="report",
                    uri=f"hf://leopardi-ocr-reports/{experiment_id}/{stage.stage}",
                    local_path=plan.report_stub_path,
                    persistence_status="queued",
                ),
            ],
            notes=[
                "Finetune control-plane artifacts materialized successfully.",
                "Use the generated plan to keep SFT and RLVR runs aligned on the rented RTX 5090 machine.",
            ],
        ),
        layout=layout,
    )
    return {
        "layout": layout.as_dict(),
        "plan": asdict(plan),
    }
