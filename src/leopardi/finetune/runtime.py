from __future__ import annotations

import torch
from torch import nn

from leopardi.finetune.config import FinetuneStageConfig


def finetune_effective_batch_size(stage_config: FinetuneStageConfig) -> int:
    return (
        stage_config.runtime.micro_batch_size * stage_config.runtime.gradient_accumulation_steps
    )


def build_finetune_optimizer(
    model: nn.Module,
    stage_config: FinetuneStageConfig,
) -> torch.optim.Optimizer:
    optimizer_cfg = stage_config.optimizer
    if optimizer_cfg.name.lower() != "adamw":
        raise ValueError(f"Unsupported finetune optimizer: {optimizer_cfg.name}")
    return torch.optim.AdamW(
        model.parameters(),
        lr=optimizer_cfg.lr,
        betas=optimizer_cfg.betas,
        eps=optimizer_cfg.eps,
        weight_decay=optimizer_cfg.weight_decay,
    )


def apply_finetune_runtime_policy(model: nn.Module, stage_config: FinetuneStageConfig) -> nn.Module:
    model.train()
    if stage_config.runtime.compile_model:
        model = torch.compile(model)
    return model
