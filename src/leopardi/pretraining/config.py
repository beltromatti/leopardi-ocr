from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml


@dataclass(slots=True)
class OptimizerConfig:
    name: str = "adamw"
    lr: float = 3e-4
    betas: tuple[float, float] = (0.9, 0.95)
    weight_decay: float = 0.1
    eps: float = 1e-8
    grad_clip_norm: float = 1.0


@dataclass(slots=True)
class RuntimeConfig:
    hardware_tag: str = "rtx5090"
    precision: str = "bf16"
    compile_model: bool = False
    gradient_checkpointing: bool = True
    micro_batch_size: int = 1
    gradient_accumulation_steps: int = 16
    dataloader_workers: int = 4
    max_steps: int = 50_000
    log_every: int = 20
    eval_every: int = 1_000
    save_every: int = 1_000


@dataclass(slots=True)
class ObjectiveWeights:
    token_ce: float = 1.0
    block_type: float = 0.2
    block_length: float = 0.1
    specialist_hint: float = 0.1
    block_box: float = 0.05
    rotation: float = 0.05
    handwriting: float = 0.05
    formula_tokens: float = 0.1
    table_blocks: float = 0.1
    table_spans: float = 0.05


@dataclass(slots=True)
class PretrainStageConfig:
    stage: str
    track: str = "s0-core"
    text_only: bool = False
    visual_mode: str = "standard"
    optimizer: OptimizerConfig = field(default_factory=OptimizerConfig)
    runtime: RuntimeConfig = field(default_factory=RuntimeConfig)
    objective_weights: ObjectiveWeights = field(default_factory=ObjectiveWeights)

    @classmethod
    def from_dict(cls, payload: dict[str, Any], runtime_payload: dict[str, Any] | None = None) -> "PretrainStageConfig":
        runtime_root = runtime_payload.get("runtime", runtime_payload or {})
        objective = payload.get("objective_weights", payload.get("loss_weights", {}))
        return cls(
            stage=payload["stage"],
            track=payload.get("track", "s0-core"),
            text_only=payload.get("text_only", False),
            visual_mode=payload.get("visual_mode", "standard"),
            optimizer=OptimizerConfig(**payload.get("optimizer", {})),
            runtime=RuntimeConfig(
                hardware_tag=runtime_root.get("hardware_tag", "rtx5090"),
                precision=runtime_root.get("precision", "bf16"),
                compile_model=runtime_root.get("compile_model", False),
                gradient_checkpointing=runtime_root.get("gradient_checkpointing", True),
                micro_batch_size=runtime_root.get("micro_batch_size", 1),
                gradient_accumulation_steps=runtime_root.get("gradient_accumulation_steps", 16),
                dataloader_workers=runtime_root.get("dataloader_workers", 4),
                max_steps=runtime_root.get("max_steps", payload.get("max_steps", 50_000)),
                log_every=runtime_root.get("log_every", 20),
                eval_every=runtime_root.get("eval_every", 1_000),
                save_every=runtime_root.get("save_every", 1_000),
            ),
            objective_weights=ObjectiveWeights(**objective),
        )

    @classmethod
    def from_yaml(
        cls,
        stage_path: str | Path,
        runtime_path: str | Path | None = None,
    ) -> "PretrainStageConfig":
        with Path(stage_path).open("r", encoding="utf-8") as handle:
            stage_payload = yaml.safe_load(handle)
        runtime_payload: dict[str, Any] | None = None
        if runtime_path is not None:
            with Path(runtime_path).open("r", encoding="utf-8") as handle:
                runtime_payload = yaml.safe_load(handle)
        return cls.from_dict(stage_payload, runtime_payload=runtime_payload)
