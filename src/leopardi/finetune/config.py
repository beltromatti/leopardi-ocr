from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml


@dataclass(slots=True)
class FinetuneOptimizerConfig:
    name: str = "adamw"
    lr: float = 1e-4
    betas: tuple[float, float] = (0.9, 0.95)
    weight_decay: float = 0.05
    eps: float = 1e-8
    grad_clip_norm: float = 1.0


@dataclass(slots=True)
class FinetuneRuntimeConfig:
    hardware_tag: str = "rtx5090"
    precision: str = "bf16"
    compile_model: bool = False
    gradient_checkpointing: bool = True
    micro_batch_size: int = 1
    gradient_accumulation_steps: int = 16
    dataloader_workers: int = 4
    max_steps: int = 12_000
    log_every: int = 10
    eval_every: int = 250
    save_every: int = 250


@dataclass(slots=True)
class AdapterConfig:
    mode: str = "full"
    rank: int = 16
    alpha: int = 32
    dropout: float = 0.05
    target_modules: tuple[str, ...] = ("writer", "planner")


@dataclass(slots=True)
class RewardWeights:
    markdown_validity: float = 1.0
    latex_validity: float = 0.8
    table_validity: float = 0.8
    reading_order: float = 0.4
    edit_similarity: float = 1.2
    output_length_penalty: float = 0.1
    latency_penalty: float = 0.2
    repair_budget_penalty: float = 0.15


@dataclass(slots=True)
class FinetuneLossWeights:
    token_ce: float = 1.0
    repair_ce: float = 0.5
    block_type: float = 0.15
    block_length: float = 0.05
    specialist_hint: float = 0.1
    block_box: float = 0.05
    rotation: float = 0.05
    handwriting: float = 0.05
    formula_tokens: float = 0.1
    table_blocks: float = 0.1
    table_spans: float = 0.05
    reward_anchor: float = 0.0


@dataclass(slots=True)
class FinetuneStageConfig:
    stage: str
    track: str = "s0-core"
    visual_mode: str = "standard"
    text_only: bool = False
    adapter: AdapterConfig = field(default_factory=AdapterConfig)
    optimizer: FinetuneOptimizerConfig = field(default_factory=FinetuneOptimizerConfig)
    runtime: FinetuneRuntimeConfig = field(default_factory=FinetuneRuntimeConfig)
    loss_weights: FinetuneLossWeights = field(default_factory=FinetuneLossWeights)
    reward_weights: RewardWeights = field(default_factory=RewardWeights)

    @classmethod
    def from_dict(
        cls,
        payload: dict[str, Any],
        runtime_payload: dict[str, Any] | None = None,
    ) -> "FinetuneStageConfig":
        runtime_root = (runtime_payload or {}).get("runtime", runtime_payload or {})
        adapter_payload = payload.get("adapter", {})
        return cls(
            stage=payload["stage"],
            track=payload.get("track", "s0-core"),
            visual_mode=payload.get("visual_mode", "standard"),
            text_only=payload.get("text_only", False),
            adapter=AdapterConfig(
                mode=adapter_payload.get("mode", "full"),
                rank=adapter_payload.get("rank", 16),
                alpha=adapter_payload.get("alpha", 32),
                dropout=adapter_payload.get("dropout", 0.05),
                target_modules=tuple(adapter_payload.get("target_modules", ("writer", "planner"))),
            ),
            optimizer=FinetuneOptimizerConfig(**payload.get("optimizer", {})),
            runtime=FinetuneRuntimeConfig(
                hardware_tag=runtime_root.get("hardware_tag", "rtx5090"),
                precision=runtime_root.get("precision", "bf16"),
                compile_model=runtime_root.get("compile_model", False),
                gradient_checkpointing=runtime_root.get("gradient_checkpointing", True),
                micro_batch_size=runtime_root.get("micro_batch_size", 1),
                gradient_accumulation_steps=runtime_root.get("gradient_accumulation_steps", 16),
                dataloader_workers=runtime_root.get("dataloader_workers", 4),
                max_steps=runtime_root.get("max_steps", payload.get("max_steps", 12_000)),
                log_every=runtime_root.get("log_every", 10),
                eval_every=runtime_root.get("eval_every", 250),
                save_every=runtime_root.get("save_every", 250),
            ),
            loss_weights=FinetuneLossWeights(**payload.get("loss_weights", {})),
            reward_weights=RewardWeights(**payload.get("reward_weights", {})),
        )

    @classmethod
    def from_yaml(
        cls,
        stage_path: str | Path,
        runtime_path: str | Path | None = None,
    ) -> "FinetuneStageConfig":
        with Path(stage_path).open("r", encoding="utf-8") as handle:
            stage_payload = yaml.safe_load(handle)
        runtime_payload: dict[str, Any] | None = None
        if runtime_path is not None:
            with Path(runtime_path).open("r", encoding="utf-8") as handle:
                runtime_payload = yaml.safe_load(handle)
        return cls.from_dict(stage_payload, runtime_payload=runtime_payload)
