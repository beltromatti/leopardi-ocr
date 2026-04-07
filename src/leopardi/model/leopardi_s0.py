from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch
from torch import Tensor, nn
from torch.nn import functional as F

from leopardi.model.config import LeopardiS0Config
from leopardi.model.modules import (
    CanonicalizedPage,
    ConvStage,
    PageCanonicalizer,
    PlannerBlock,
    CrossAttentionBlock,
    WriterBlock,
    flatten_pooled_feature_map,
    make_causal_mask,
)


@dataclass(slots=True)
class PlannerOutputs:
    states: Tensor
    block_type_logits: Tensor
    block_length_logits: Tensor
    specialist_hint_logits: Tensor
    block_boxes: Tensor
    stop_logits: Tensor
    confidence_logits: Tensor


@dataclass(slots=True)
class AuxiliaryOutputs:
    rotation_logits: Tensor
    handwriting_logits: Tensor
    formula_token_logits: Tensor
    table_block_logits: Tensor
    table_span_logits: Tensor


@dataclass(slots=True)
class LeopardiS0Output:
    canonicalized: CanonicalizedPage
    visual_tokens: Tensor
    structural_latents: Tensor
    planner: PlannerOutputs
    decoder_logits: Tensor
    auxiliary: AuxiliaryOutputs


class AdaptiveVisualTokenizer(nn.Module):
    def __init__(self, config: LeopardiS0Config) -> None:
        super().__init__()
        vision = config.visual_tokenizer
        self.config = vision
        self.stem = nn.Sequential(
            nn.Conv2d(3, vision.stem_dim, kernel_size=7, stride=4, padding=3),
            nn.GELU(),
        )
        dims = [vision.stem_dim, *vision.stage_dims]
        self.stages = nn.ModuleList(
            [
                ConvStage(dims[i], dims[i + 1], depth=vision.stage_depths[i], dropout=vision.dropout)
                for i in range(len(vision.stage_dims))
            ]
        )
        self.projections = nn.ModuleList(
            [
                nn.Linear(vision.stage_dims[index], vision.hidden_size)
                for index in vision.stage_indices
            ]
        )
        self.final_norm = nn.LayerNorm(vision.hidden_size)

    def resolve_layout(self, mode_or_budget: str | int | None) -> list[list[int]]:
        if mode_or_budget is None:
            return self.config.pool_layouts["standard"]
        if isinstance(mode_or_budget, str):
            return self.config.pool_layouts.get(mode_or_budget, self.config.pool_layouts["standard"])

        budget = int(mode_or_budget)
        if budget <= 80:
            return self.config.pool_layouts["fast"]
        if budget >= 160:
            return self.config.pool_layouts["hard"]
        return self.config.pool_layouts["standard"]

    def forward(self, image: Tensor, mode_or_budget: str | int | None = None) -> tuple[Tensor, Tensor]:
        x = self.stem(image)
        feature_maps: list[Tensor] = []
        for stage in self.stages:
            x = stage(x)
            feature_maps.append(x)

        selected_maps = [feature_maps[index] for index in self.config.stage_indices]
        layout = self.resolve_layout(mode_or_budget)
        tokens = []
        for feature_map, grid, projection in zip(selected_maps, layout, self.projections, strict=True):
            flattened = flatten_pooled_feature_map(feature_map, grid)
            tokens.append(projection(flattened))

        visual_tokens = self.final_norm(torch.cat(tokens, dim=1))
        page_summary = visual_tokens.mean(dim=1)
        return visual_tokens, page_summary


class StructuralLatentBottleneck(nn.Module):
    def __init__(self, config: LeopardiS0Config) -> None:
        super().__init__()
        latent_cfg = config.latent_bottleneck
        self.latents = nn.Parameter(torch.randn(latent_cfg.num_latents, latent_cfg.hidden_size) * 0.02)
        self.blocks = nn.ModuleList(
            [
                CrossAttentionBlock(
                    latent_cfg.hidden_size,
                    latent_cfg.num_heads,
                    mlp_ratio=latent_cfg.mlp_ratio,
                    dropout=latent_cfg.dropout,
                )
                for _ in range(latent_cfg.num_layers)
            ]
        )

    def forward(self, visual_tokens: Tensor) -> Tensor:
        latents = self.latents.unsqueeze(0).expand(visual_tokens.size(0), -1, -1)
        for block in self.blocks:
            latents = block(latents, visual_tokens)
        return latents


class BlockPlanner(nn.Module):
    def __init__(self, config: LeopardiS0Config) -> None:
        super().__init__()
        planner_cfg = config.planner
        self.queries = nn.Parameter(torch.randn(planner_cfg.num_blocks, planner_cfg.hidden_size) * 0.02)
        self.blocks = nn.ModuleList(
            [
                PlannerBlock(
                    planner_cfg.hidden_size,
                    planner_cfg.num_heads,
                    dropout=planner_cfg.dropout,
                )
                for _ in range(planner_cfg.num_layers)
            ]
        )
        self.block_type_head = nn.Linear(planner_cfg.hidden_size, len(planner_cfg.block_types))
        self.length_head = nn.Linear(planner_cfg.hidden_size, planner_cfg.num_length_buckets)
        self.hint_head = nn.Linear(planner_cfg.hidden_size, len(planner_cfg.specialist_hints))
        self.box_head = nn.Linear(planner_cfg.hidden_size, 4)
        self.stop_head = nn.Linear(planner_cfg.hidden_size, 1)
        self.confidence_head = nn.Linear(planner_cfg.hidden_size, 1)

    def forward(self, structural_latents: Tensor) -> PlannerOutputs:
        states = self.queries.unsqueeze(0).expand(structural_latents.size(0), -1, -1)
        for block in self.blocks:
            states = block(states, structural_latents)
        return PlannerOutputs(
            states=states,
            block_type_logits=self.block_type_head(states),
            block_length_logits=self.length_head(states),
            specialist_hint_logits=self.hint_head(states),
            block_boxes=torch.sigmoid(self.box_head(states)),
            stop_logits=self.stop_head(states).squeeze(-1),
            confidence_logits=self.confidence_head(states).squeeze(-1),
        )


class WriterDecoder(nn.Module):
    def __init__(self, config: LeopardiS0Config) -> None:
        super().__init__()
        decoder_cfg = config.writer_decoder
        self.token_embedding = nn.Embedding(decoder_cfg.vocab_size, decoder_cfg.hidden_size)
        self.position_embedding = nn.Embedding(decoder_cfg.max_seq_len, decoder_cfg.hidden_size)
        self.dropout = nn.Dropout(decoder_cfg.dropout)
        self.blocks = nn.ModuleList(
            [
                WriterBlock(
                    decoder_cfg.hidden_size,
                    decoder_cfg.num_heads,
                    mlp_ratio=decoder_cfg.mlp_ratio,
                    dropout=decoder_cfg.dropout,
                )
                for _ in range(decoder_cfg.num_layers)
            ]
        )
        self.norm = nn.LayerNorm(decoder_cfg.hidden_size)
        self.tie_embeddings = decoder_cfg.tie_embeddings
        if not self.tie_embeddings:
            self.output_projection = nn.Linear(decoder_cfg.hidden_size, decoder_cfg.vocab_size)

    def forward(self, input_ids: Tensor, memory: Tensor) -> Tensor:
        batch, seq_len = input_ids.shape
        positions = torch.arange(seq_len, device=input_ids.device).unsqueeze(0).expand(batch, -1)
        x = self.token_embedding(input_ids) + self.position_embedding(positions)
        x = self.dropout(x)
        causal_mask = make_causal_mask(seq_len, input_ids.device)
        for block in self.blocks:
            x = block(x, memory, causal_mask=causal_mask)
        x = self.norm(x)
        if self.tie_embeddings:
            return F.linear(x, self.token_embedding.weight)
        return self.output_projection(x)


class LeopardiS0(nn.Module):
    def __init__(self, config: LeopardiS0Config) -> None:
        super().__init__()
        self.config = config
        self.canonicalizer = PageCanonicalizer(
            line_density_pool=config.page_canonicalizer.line_density_pool,
            contrast_eps=config.page_canonicalizer.contrast_eps,
        )
        self.visual_tokenizer = AdaptiveVisualTokenizer(config)
        self.latent_bottleneck = StructuralLatentBottleneck(config)
        self.planner = BlockPlanner(config)
        self.writer = WriterDecoder(config)

        hidden = config.hidden_size
        self.rotation_head = nn.Linear(hidden, config.auxiliary_heads.rotation_classes)
        self.handwriting_head = nn.Linear(hidden, config.auxiliary_heads.handwriting_classes)
        self.formula_token_head = nn.Linear(hidden, 1)
        self.table_block_head = nn.Linear(hidden, 1)
        self.table_span_head = nn.Linear(hidden, config.auxiliary_heads.table_span_dims)

    @classmethod
    def from_yaml(cls, path: str) -> "LeopardiS0":
        return cls(LeopardiS0Config.from_yaml(path))

    def forward(
        self,
        image: Tensor,
        decoder_input_ids: Tensor,
        visual_mode: str | int | None = None,
    ) -> LeopardiS0Output:
        canonicalized = self.canonicalizer(image)
        visual_tokens, page_summary = self.visual_tokenizer(canonicalized.image, visual_mode)
        structural_latents = self.latent_bottleneck(visual_tokens)
        planner_outputs = self.planner(structural_latents)
        memory = torch.cat((structural_latents, planner_outputs.states), dim=1)
        decoder_logits = self.writer(decoder_input_ids, memory)

        auxiliary = AuxiliaryOutputs(
            rotation_logits=self.rotation_head(page_summary),
            handwriting_logits=self.handwriting_head(page_summary),
            formula_token_logits=self.formula_token_head(visual_tokens).squeeze(-1),
            table_block_logits=self.table_block_head(planner_outputs.states).squeeze(-1),
            table_span_logits=self.table_span_head(planner_outputs.states),
        )
        return LeopardiS0Output(
            canonicalized=canonicalized,
            visual_tokens=visual_tokens,
            structural_latents=structural_latents,
            planner=planner_outputs,
            decoder_logits=decoder_logits,
            auxiliary=auxiliary,
        )

    def num_parameters(self, trainable_only: bool = True) -> int:
        params = self.parameters() if not trainable_only else (p for p in self.parameters() if p.requires_grad)
        return sum(parameter.numel() for parameter in params)

    def summary(self) -> dict[str, Any]:
        total_params = self.num_parameters()
        return {
            "family": self.config.family,
            "target_params_m": self.config.target_params_m,
            "actual_params_m": round(total_params / 1_000_000, 2),
            "hidden_size": self.config.hidden_size,
            "num_latents": self.config.latent_bottleneck.num_latents,
            "num_planner_blocks": self.config.planner.num_blocks,
            "decoder_layers": self.config.writer_decoder.num_layers,
            "vocab_size": self.config.writer_decoder.vocab_size,
        }
