from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor

from leopardi.finetune.config import FinetuneStageConfig


@dataclass(slots=True)
class RewardBreakdown:
    total_reward: Tensor
    reward_terms: dict[str, float]


def compute_reward_breakdown(
    signals: dict[str, Tensor],
    stage_config: FinetuneStageConfig,
) -> RewardBreakdown:
    weights = stage_config.reward_weights
    term_map = {
        "markdown_validity": weights.markdown_validity,
        "latex_validity": weights.latex_validity,
        "table_validity": weights.table_validity,
        "reading_order": weights.reading_order,
        "edit_similarity": weights.edit_similarity,
        "output_length_penalty": -abs(weights.output_length_penalty),
        "latency_penalty": -abs(weights.latency_penalty),
        "repair_budget_penalty": -abs(weights.repair_budget_penalty),
    }
    terms: dict[str, Tensor] = {}
    for name, weight in term_map.items():
        if name in signals:
            terms[name] = signals[name].mean() * weight
    if not terms:
        zero = torch.tensor(0.0)
        return RewardBreakdown(total_reward=zero, reward_terms={})
    total = torch.stack(list(terms.values())).sum()
    return RewardBreakdown(
        total_reward=total,
        reward_terms={name: round(value.detach().item(), 6) for name, value in terms.items()},
    )
