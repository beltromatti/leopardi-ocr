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
    seed: int = 1337


@dataclass(slots=True)
class SchedulerConfig:
    name: str = "cosine"
    warmup_ratio: float = 0.05
    min_lr_ratio: float = 0.1


@dataclass(slots=True)
class AdapterConfig:
    mode: str = "full"
    rank: int = 16
    alpha: int = 32
    dropout: float = 0.05
    target_modules: tuple[str, ...] = ("writer", "planner")


@dataclass(slots=True)
class SamplingConfig:
    easy_ratio: float = 0.20
    medium_ratio: float = 0.30
    hard_ratio: float = 0.35
    pathological_ratio: float = 0.15
    failure_replay_ratio: float = 0.20
    refresh_failure_buffer_every: int = 250
    keep_clean_anchor_fraction: float = 0.10


@dataclass(slots=True)
class ModuleLrConfig:
    visual_tokenizer: float = 0.5
    layout_side_encoder: float = 0.6
    latent_bottleneck: float = 0.8
    planner: float = 1.1
    writer: float = 1.25
    auxiliary_heads: float = 1.0
    no_decay_on_norms_and_bias: bool = True


@dataclass(slots=True)
class VerifierConfig:
    reward_clip: float = 2.5
    normalize_rewards: bool = True
    min_reward_group_size: int = 4
    informative_reward_floor: float = 0.02
    target_latency_ms: float = 1200.0
    target_output_tokens: int = 3072
    kl_anchor_beta: float = 0.02


@dataclass(slots=True)
class RewardWeights:
    markdown_validity: float = 1.0
    latex_validity: float = 0.8
    table_validity: float = 0.8
    reading_order: float = 0.4
    edit_similarity: float = 1.2
    formula_exactness: float = 0.8
    header_footer_suppression: float = 0.4
    chart_text: float = 0.4
    output_length_penalty: float = 0.1
    latency_penalty: float = 0.2
    repair_budget_penalty: float = 0.15


@dataclass(slots=True)
class FinetuneLossWeights:
    token_ce: float = 1.0
    mtp_ce: float = 0.1
    formula_ce: float = 0.15
    table_ce: float = 0.15
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
    kl_anchor: float = 0.0
    label_smoothing: float = 0.0
    sample_weight_floor: float = 0.25


@dataclass(slots=True)
class FinetuneStageConfig:
    stage: str
    track: str = "s0-core"
    visual_mode: str = "standard"
    text_only: bool = False
    data_bundle_ids: tuple[str, ...] = ("f0_general_sft_v1",)
    adapter: AdapterConfig = field(default_factory=AdapterConfig)
    optimizer: FinetuneOptimizerConfig = field(default_factory=FinetuneOptimizerConfig)
    runtime: FinetuneRuntimeConfig = field(default_factory=FinetuneRuntimeConfig)
    scheduler: SchedulerConfig = field(default_factory=SchedulerConfig)
    sampling: SamplingConfig = field(default_factory=SamplingConfig)
    module_lr: ModuleLrConfig = field(default_factory=ModuleLrConfig)
    verifier: VerifierConfig = field(default_factory=VerifierConfig)
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
            data_bundle_ids=tuple(payload.get("data_bundle_ids", ("f0_general_sft_v1",))),
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
                seed=runtime_root.get("seed", payload.get("seed", 1337)),
            ),
            scheduler=SchedulerConfig(**payload.get("scheduler", {})),
            sampling=SamplingConfig(**payload.get("sampling", {})),
            module_lr=ModuleLrConfig(**payload.get("module_lr", {})),
            verifier=VerifierConfig(**payload.get("verifier", {})),
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
