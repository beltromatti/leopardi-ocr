from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor


@dataclass(slots=True)
class PretrainBatch:
    image: Tensor
    decoder_input_ids: Tensor
    labels: Tensor
    label_mask: Tensor | None = None
    sample_weights: Tensor | None = None
    formula_label_mask: Tensor | None = None
    table_label_mask: Tensor | None = None
    block_types: Tensor | None = None
    block_lengths: Tensor | None = None
    specialist_hints: Tensor | None = None
    block_boxes: Tensor | None = None
    block_mask: Tensor | None = None
    planner_confidence_targets: Tensor | None = None
    rotation_labels: Tensor | None = None
    handwriting_labels: Tensor | None = None
    formula_token_mask: Tensor | None = None
    table_block_mask: Tensor | None = None
    table_span_targets: Tensor | None = None

    @classmethod
    def synthetic(
        cls,
        batch_size: int,
        image_size: tuple[int, int] = (256, 256),
        seq_len: int = 64,
        vocab_size: int = 512,
        planner_blocks: int = 12,
        visual_tokens: int = 25,
        num_block_types: int = 10,
        num_length_buckets: int = 4,
        num_hints: int = 5,
        rotation_classes: int = 4,
        handwriting_classes: int = 3,
    ) -> "PretrainBatch":
        height, width = image_size
        device = torch.device("cpu")
        return cls(
            image=torch.rand(batch_size, 3, height, width, device=device),
            decoder_input_ids=torch.randint(0, vocab_size, (batch_size, seq_len), device=device),
            labels=torch.randint(0, vocab_size, (batch_size, seq_len), device=device),
            label_mask=torch.ones(batch_size, seq_len, dtype=torch.bool, device=device),
            sample_weights=torch.rand(batch_size, device=device) * 0.4 + 0.8,
            formula_label_mask=torch.randint(0, 2, (batch_size, seq_len), device=device).float(),
            table_label_mask=torch.randint(0, 2, (batch_size, seq_len), device=device).float(),
            block_types=torch.randint(0, num_block_types, (batch_size, planner_blocks), device=device),
            block_lengths=torch.randint(0, num_length_buckets, (batch_size, planner_blocks), device=device),
            specialist_hints=torch.randint(0, num_hints, (batch_size, planner_blocks), device=device),
            block_boxes=torch.rand(batch_size, planner_blocks, 4, device=device),
            block_mask=torch.ones(batch_size, planner_blocks, dtype=torch.bool, device=device),
            planner_confidence_targets=torch.rand(batch_size, planner_blocks, device=device),
            rotation_labels=torch.randint(0, rotation_classes, (batch_size,), device=device),
            handwriting_labels=torch.randint(0, handwriting_classes, (batch_size,), device=device),
            formula_token_mask=torch.randint(0, 2, (batch_size, visual_tokens), device=device).float(),
            table_block_mask=torch.randint(0, 2, (batch_size, planner_blocks), device=device).float(),
            table_span_targets=torch.rand(batch_size, planner_blocks, 4, device=device),
        )
