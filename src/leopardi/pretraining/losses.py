from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor
from torch.nn import functional as F

from leopardi.model.leopardi_s0 import LeopardiS0Output
from leopardi.pretraining.batch import PretrainBatch
from leopardi.pretraining.config import PretrainStageConfig


@dataclass(slots=True)
class LossReport:
    total_loss: Tensor
    loss_terms: dict[str, float]


def _masked_cross_entropy(logits: Tensor, targets: Tensor, mask: Tensor | None = None) -> Tensor:
    return _weighted_cross_entropy(logits, targets, mask=mask)


def _weighted_cross_entropy(
    logits: Tensor,
    targets: Tensor,
    mask: Tensor | None = None,
    sample_weights: Tensor | None = None,
    label_smoothing: float = 0.0,
) -> Tensor:
    flat_loss = F.cross_entropy(
        logits.reshape(-1, logits.size(-1)),
        targets.reshape(-1),
        reduction="none",
        label_smoothing=label_smoothing,
    ).reshape_as(targets)
    weight_map = torch.ones_like(flat_loss)
    if mask is not None:
        weight_map = weight_map * mask.float()
    if sample_weights is not None:
        shape = (sample_weights.shape[0],) + (1,) * (flat_loss.dim() - 1)
        weight_map = weight_map * sample_weights.view(shape)
    denom = weight_map.sum().clamp_min(1.0)
    return (flat_loss * weight_map).sum() / denom


def _multi_token_prediction_loss(
    mtp_logits: tuple[Tensor, ...] | None,
    targets: Tensor,
    mask: Tensor | None = None,
    sample_weights: Tensor | None = None,
    label_smoothing: float = 0.0,
) -> Tensor | None:
    if not mtp_logits:
        return None

    losses: list[Tensor] = []
    for step, future_logits in enumerate(mtp_logits, start=1):
        if future_logits.size(1) <= step or targets.size(1) <= step:
            continue
        future_targets = targets[:, step:]
        aligned_logits = future_logits[:, :-step, :]
        future_mask = mask[:, step:] if mask is not None else None
        losses.append(
            _weighted_cross_entropy(
                aligned_logits,
                future_targets,
                mask=future_mask,
                sample_weights=sample_weights,
                label_smoothing=label_smoothing,
            )
        )
    if not losses:
        return None
    return torch.stack(losses).mean()


def compute_pretraining_losses(
    outputs: LeopardiS0Output,
    batch: PretrainBatch,
    stage_config: PretrainStageConfig,
) -> LossReport:
    weights = stage_config.objective_weights
    terms: dict[str, Tensor] = {}
    sample_weights = None
    if batch.sample_weights is not None:
        sample_weights = batch.sample_weights.clamp_min(weights.sample_weight_floor)

    token_loss = _weighted_cross_entropy(
        outputs.decoder_logits,
        batch.labels,
        mask=batch.label_mask,
        sample_weights=sample_weights,
        label_smoothing=weights.label_smoothing,
    )
    terms["token_ce"] = token_loss * weights.token_ce
    mtp_loss = _multi_token_prediction_loss(
        outputs.mtp_logits,
        batch.labels,
        mask=batch.label_mask,
        sample_weights=sample_weights,
        label_smoothing=weights.label_smoothing,
    )
    if mtp_loss is not None and weights.mtp_ce > 0:
        terms["mtp_ce"] = mtp_loss * weights.mtp_ce

    if batch.formula_label_mask is not None and weights.formula_ce > 0:
        terms["formula_ce"] = _weighted_cross_entropy(
            outputs.decoder_logits,
            batch.labels,
            mask=batch.formula_label_mask,
            sample_weights=sample_weights,
            label_smoothing=weights.label_smoothing,
        ) * weights.formula_ce
    if batch.table_label_mask is not None and weights.table_ce > 0:
        terms["table_ce"] = _weighted_cross_entropy(
            outputs.decoder_logits,
            batch.labels,
            mask=batch.table_label_mask,
            sample_weights=sample_weights,
            label_smoothing=weights.label_smoothing,
        ) * weights.table_ce

    if batch.block_types is not None:
        terms["block_type"] = _masked_cross_entropy(
            outputs.planner.block_type_logits, batch.block_types, batch.block_mask
        ) * weights.block_type
    if batch.block_lengths is not None:
        terms["block_length"] = _masked_cross_entropy(
            outputs.planner.block_length_logits, batch.block_lengths, batch.block_mask
        ) * weights.block_length
    if batch.specialist_hints is not None:
        terms["specialist_hint"] = _masked_cross_entropy(
            outputs.planner.specialist_hint_logits, batch.specialist_hints, batch.block_mask
        ) * weights.specialist_hint
    if batch.block_boxes is not None:
        l1 = F.l1_loss(outputs.planner.block_boxes, batch.block_boxes, reduction="none").mean(dim=-1)
        if batch.block_mask is not None:
            l1 = (l1 * batch.block_mask.float()).sum() / batch.block_mask.float().sum().clamp_min(1.0)
        else:
            l1 = l1.mean()
        terms["block_box"] = l1 * weights.block_box
    if batch.planner_confidence_targets is not None and weights.planner_confidence > 0:
        confidence_loss = F.binary_cross_entropy_with_logits(
            outputs.planner.confidence_logits,
            batch.planner_confidence_targets,
        )
        terms["planner_confidence"] = confidence_loss * weights.planner_confidence
    if batch.rotation_labels is not None:
        terms["rotation"] = F.cross_entropy(
            outputs.auxiliary.rotation_logits, batch.rotation_labels
        ) * weights.rotation
    if batch.handwriting_labels is not None:
        terms["handwriting"] = F.cross_entropy(
            outputs.auxiliary.handwriting_logits, batch.handwriting_labels
        ) * weights.handwriting
    if batch.formula_token_mask is not None:
        terms["formula_tokens"] = F.binary_cross_entropy_with_logits(
            outputs.auxiliary.formula_token_logits, batch.formula_token_mask
        ) * weights.formula_tokens
    if batch.table_block_mask is not None:
        terms["table_blocks"] = F.binary_cross_entropy_with_logits(
            outputs.auxiliary.table_block_logits, batch.table_block_mask
        ) * weights.table_blocks
    if batch.table_span_targets is not None:
        terms["table_spans"] = F.l1_loss(
            outputs.auxiliary.table_span_logits, batch.table_span_targets
        ) * weights.table_spans

    total = torch.stack([value for value in terms.values()]).sum()
    return LossReport(
        total_loss=total,
        loss_terms={name: round(value.detach().item(), 6) for name, value in terms.items()},
    )
