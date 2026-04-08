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
    verifier = stage_config.verifier
    term_map = {
        "markdown_validity": weights.markdown_validity,
        "latex_validity": weights.latex_validity,
        "table_validity": weights.table_validity,
        "reading_order": weights.reading_order,
        "edit_similarity": weights.edit_similarity,
        "formula_exactness": weights.formula_exactness,
        "header_footer_suppression": weights.header_footer_suppression,
        "chart_text": weights.chart_text,
        "output_length_penalty": -abs(weights.output_length_penalty),
        "latency_penalty": -abs(weights.latency_penalty),
        "repair_budget_penalty": -abs(weights.repair_budget_penalty),
    }
    terms: dict[str, Tensor] = {}
    for name, weight in term_map.items():
        if name in signals:
            value = signals[name]
            if verifier.normalize_rewards and value.numel() >= verifier.min_reward_group_size:
                value = (value - value.mean()) / value.std(unbiased=False).clamp_min(1e-6)
            value = value.clamp(min=-verifier.reward_clip, max=verifier.reward_clip)
            averaged = value.mean()
            if averaged.abs().item() < verifier.informative_reward_floor:
                continue
            terms[name] = averaged * weight
    if not terms:
        zero = torch.tensor(0.0)
        return RewardBreakdown(total_reward=zero, reward_terms={})
    total = torch.stack(list(terms.values())).sum()
    return RewardBreakdown(
        total_reward=total,
        reward_terms={name: round(value.detach().item(), 6) for name, value in terms.items()},
    )
