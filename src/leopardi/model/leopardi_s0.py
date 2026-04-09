"""Leopardi-S0: ~150M document parser with pretrained SigLIP2 vision encoder
and SmolLM2-initialized writer decoder.

All forward paths are torch.compile-friendly.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch
from torch import Tensor, nn
from torch.nn import functional as F

from leopardi.model.config import LeopardiS0Config
from leopardi.model.modules import (
    BottleneckBlock,
    CanonicalizedPage,
    PageCanonicalizer,
    PlannerBlock,
    RMSNorm,
    WriterBlock,
    _build_rope_cache,
    make_causal_mask,
    pixel_shuffle_down,
)


def _evenly_spaced_indices(source_count: int, target_count: int) -> list[int]:
    if target_count <= 0:
        return []
    if target_count >= source_count:
        return list(range(source_count))
    indices = torch.linspace(0, source_count - 1, steps=target_count).round().to(torch.int64)
    deduped: list[int] = []
    for value in indices.tolist():
        if not deduped or deduped[-1] != value:
            deduped.append(value)
    while len(deduped) < target_count:
        candidate = min(source_count - 1, deduped[-1] + 1 if deduped else 0)
        deduped.append(candidate)
    return deduped[:target_count]


def _resample_vector(vector: Tensor, target_size: int) -> Tensor:
    if vector.numel() == target_size:
        return vector.detach().clone()
    reshaped = vector.detach().float().view(1, 1, -1)
    resized = F.interpolate(reshaped, size=target_size, mode="linear", align_corners=False)
    return resized.view(target_size).to(vector.dtype)


def _resample_matrix(weight: Tensor, target_shape: tuple[int, int]) -> Tensor:
    if tuple(weight.shape) == target_shape:
        return weight.detach().clone()
    reshaped = weight.detach().float().unsqueeze(0).unsqueeze(0)
    resized = F.interpolate(reshaped, size=target_shape, mode="bilinear", align_corners=False)
    return resized.squeeze(0).squeeze(0).to(weight.dtype)


def _resample_attention_weight(
    weight: Tensor,
    *,
    source_heads: int,
    target_heads: int,
    head_dim: int,
    target_input_dim: int,
) -> Tensor:
    source_input_dim = weight.shape[1]
    if source_heads == target_heads and source_input_dim == target_input_dim:
        return weight.detach().clone()
    reshaped = weight.detach().float().view(source_heads, head_dim, source_input_dim)
    head_indices = _evenly_spaced_indices(source_heads, target_heads)
    picked = reshaped[head_indices]
    flattened = picked.reshape(target_heads * head_dim, 1, source_input_dim)
    resized = F.interpolate(flattened, size=target_input_dim, mode="linear", align_corners=False)
    return resized.view(target_heads * head_dim, target_input_dim).to(weight.dtype)


def _copy_parameter_(parameter: nn.Parameter, value: Tensor) -> None:
    with torch.no_grad():
        parameter.copy_(value.to(device=parameter.device, dtype=parameter.dtype))


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
    layout_tokens: Tensor
    structural_latents: Tensor
    planner: PlannerOutputs
    decoder_logits: Tensor
    mtp_logits: tuple[Tensor, ...] | None
    auxiliary: AuxiliaryOutputs


# ---------------------------------------------------------------------------
# Vision Encoder wrapper around SigLIP2
# ---------------------------------------------------------------------------

class VisionEncoder(nn.Module):
    """Wraps a pretrained SigLIP2 vision encoder with pixel-shuffle compression
    and linear projection to internal hidden dimension."""

    def __init__(self, config: LeopardiS0Config) -> None:
        super().__init__()
        vis = config.vision_encoder
        self.freeze_layers = vis.freeze_layers
        self.pixel_shuffle_factor = vis.pixel_shuffle_factor
        self.pretrained_model_name = vis.pretrained_model

        # Placeholder for the pretrained encoder — loaded externally via
        # load_pretrained_vision_encoder() to keep __init__ lightweight and
        # to support the tiny() config path without requiring transformers.
        self.encoder: nn.Module | None = None

        # Projection from SigLIP2 output to internal hidden dimension.
        # After pixel shuffle (factor=2), channel dim = output_dim * factor^2
        proj_input_dim = vis.output_dim * (vis.pixel_shuffle_factor ** 2)
        self.projection = nn.Sequential(
            nn.Linear(proj_input_dim, vis.projection_dim),
            RMSNorm(vis.projection_dim),
        )
        self.output_dim = vis.output_dim
        self.projection_dim = vis.projection_dim
        self.pretrained_loaded = False

    def load_pretrained_vision_encoder(self, *, local_files_only: bool = False) -> None:
        """Load the pretrained SigLIP2 vision model from HuggingFace."""
        if not self.pretrained_model_name:
            return
        try:
            from transformers import Siglip2VisionModel
        except ImportError:
            from transformers import SiglipVisionModel as Siglip2VisionModel

        self.encoder = Siglip2VisionModel.from_pretrained(
            self.pretrained_model_name,
            local_files_only=local_files_only,
        )
        if self.freeze_layers > 0 and hasattr(self.encoder, "vision_model"):
            layers = self.encoder.vision_model.encoder.layers
            for layer in layers[: self.freeze_layers]:
                for param in layer.parameters():
                    param.requires_grad = False
        self.pretrained_loaded = True

    def forward(self, pixel_values: Tensor) -> tuple[Tensor, Tensor]:
        if self.encoder is not None:
            with torch.set_grad_enabled(self.training):
                vision_out = self.encoder(pixel_values=pixel_values)
                hidden = vision_out.last_hidden_state  # (B, seq, output_dim)
        else:
            # Fallback for tiny config or testing without pretrained weights.
            B = pixel_values.size(0)
            hidden = torch.zeros(B, 16, self.output_dim, device=pixel_values.device, dtype=pixel_values.dtype)

        # Reshape to spatial grid for pixel shuffle.
        B, S, D = hidden.shape
        H = W = int(S ** 0.5)
        if H * W < S:
            H = W = H + 1
        # Pad if needed.
        if H * W > S:
            hidden = F.pad(hidden, (0, 0, 0, H * W - S))
        spatial = hidden.transpose(1, 2).reshape(B, D, H, W)

        # Pixel shuffle down: merge 2×2 patches → reduces tokens by 4×,
        # increases channels by 4×.
        if self.pixel_shuffle_factor > 1 and H >= self.pixel_shuffle_factor and W >= self.pixel_shuffle_factor:
            spatial = pixel_shuffle_down(spatial, self.pixel_shuffle_factor)

        # Flatten back to token sequence.
        tokens = spatial.flatten(2).transpose(1, 2)  # (B, S', D*factor^2)
        tokens = self.projection(tokens)  # (B, S', projection_dim)
        summary = tokens.mean(dim=1)
        return tokens, summary


# ---------------------------------------------------------------------------
# Layout Side-Map Encoder
# ---------------------------------------------------------------------------

class LayoutSideMapEncoder(nn.Module):
    def __init__(self, config: LeopardiS0Config) -> None:
        super().__init__()
        cfg = config.layout_side_encoder
        self.pool_grid = cfg.pool_grid
        self.encoder = nn.Sequential(
            nn.Conv2d(3, cfg.stem_dim, kernel_size=3, padding=1),
            nn.GELU(),
            nn.Conv2d(cfg.stem_dim, cfg.hidden_size, kernel_size=3, padding=1),
            nn.GELU(),
            nn.Dropout(cfg.dropout),
        )
        self.norm = RMSNorm(cfg.hidden_size)

    def forward(self, canonicalized: CanonicalizedPage) -> tuple[Tensor, Tensor]:
        line_density = canonicalized.line_density_map
        direction = F.interpolate(
            canonicalized.text_direction_map, size=line_density.shape[-2:],
            mode="bilinear", align_corners=False,
        )
        layout_maps = torch.cat((line_density, direction), dim=1)
        encoded = self.encoder(layout_maps)
        pooled = F.adaptive_avg_pool2d(encoded, output_size=self.pool_grid)
        tokens = pooled.flatten(2).transpose(1, 2)
        tokens = self.norm(tokens)
        summary = tokens.mean(dim=1)
        return tokens, summary


# ---------------------------------------------------------------------------
# Structural Latent Bottleneck
# ---------------------------------------------------------------------------

class StructuralLatentBottleneck(nn.Module):
    def __init__(self, config: LeopardiS0Config) -> None:
        super().__init__()
        cfg = config.latent_bottleneck
        self.latents = nn.Parameter(torch.randn(cfg.num_latents, cfg.hidden_size) * 0.02)
        self.blocks = nn.ModuleList([
            BottleneckBlock(cfg.hidden_size, cfg.num_heads, cfg.num_kv_heads, cfg.mlp_ratio, cfg.dropout)
            for _ in range(cfg.num_layers)
        ])

    def forward(self, context_tokens: Tensor) -> Tensor:
        latents = self.latents.unsqueeze(0).expand(context_tokens.size(0), -1, -1)
        for block in self.blocks:
            latents = block(latents, context_tokens)
        return latents


# ---------------------------------------------------------------------------
# Block Planner
# ---------------------------------------------------------------------------

class BlockPlanner(nn.Module):
    def __init__(self, config: LeopardiS0Config) -> None:
        super().__init__()
        cfg = config.planner
        self.queries = nn.Parameter(torch.randn(cfg.num_blocks, cfg.hidden_size) * 0.02)
        self.blocks = nn.ModuleList([
            PlannerBlock(cfg.hidden_size, cfg.num_heads, cfg.num_kv_heads, cfg.mlp_ratio, cfg.dropout)
            for _ in range(cfg.num_layers)
        ])
        self.block_type_head = nn.Linear(cfg.hidden_size, len(cfg.block_types))
        self.length_head = nn.Linear(cfg.hidden_size, cfg.num_length_buckets)
        self.hint_head = nn.Linear(cfg.hidden_size, len(cfg.specialist_hints))
        self.box_head = nn.Linear(cfg.hidden_size, 4)
        self.stop_head = nn.Linear(cfg.hidden_size, 1)
        self.confidence_head = nn.Linear(cfg.hidden_size, 1)

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


# ---------------------------------------------------------------------------
# Writer Decoder (modern: RoPE, GQA, SwiGLU, RMSNorm)
# ---------------------------------------------------------------------------

class WriterDecoder(nn.Module):
    def __init__(self, config: LeopardiS0Config) -> None:
        super().__init__()
        dec = config.writer_decoder
        self.hidden_size = dec.hidden_size
        self.token_embedding = nn.Embedding(dec.vocab_size, dec.hidden_size)
        self.dropout = nn.Dropout(dec.dropout)
        self.blocks = nn.ModuleList([
            WriterBlock(dec.hidden_size, dec.num_heads, dec.num_kv_heads, dec.mlp_ratio, dec.dropout)
            for _ in range(dec.num_layers)
        ])
        self.norm = RMSNorm(dec.hidden_size)
        self.tie_embeddings = dec.tie_embeddings
        if not self.tie_embeddings:
            self.output_projection = nn.Linear(dec.hidden_size, dec.vocab_size, bias=False)

        # RoPE cache
        head_dim = dec.hidden_size // dec.num_heads
        self.register_buffer(
            "rope_cache",
            _build_rope_cache(dec.max_seq_len, head_dim, dec.rope_theta, torch.device("cpu")),
            persistent=False,
        )

        # MTP
        mtp = config.multi_token_prediction
        self.mtp_enabled = mtp.enabled and mtp.horizon > 0
        self.mtp_horizon = mtp.horizon if self.mtp_enabled else 0
        if self.mtp_enabled:
            self.mtp_norm = RMSNorm(dec.hidden_size)
            self.mtp_dropout = nn.Dropout(mtp.dropout)
            self.mtp_heads = nn.ModuleList([
                nn.Linear(dec.hidden_size, dec.hidden_size, bias=False)
                for _ in range(self.mtp_horizon)
            ])
        self.pretrained_report: dict[str, Any] = {"initialized": False}

    def initialize_from_pretrained(
        self,
        *,
        repo_id: str,
        layer_indices: list[int] | None = None,
        local_files_only: bool = False,
    ) -> dict[str, Any]:
        from transformers import AutoModelForCausalLM

        if not repo_id:
            self.pretrained_report = {"initialized": False, "reason": "no_repo_id"}
            return self.pretrained_report

        source_model = AutoModelForCausalLM.from_pretrained(
            repo_id,
            local_files_only=local_files_only,
        )
        source = source_model.model
        source_cfg = source_model.config
        source_layers = source.layers
        target_layers = self.blocks

        if layer_indices is None:
            layer_indices = _evenly_spaced_indices(len(source_layers), len(target_layers))
        if len(layer_indices) != len(target_layers):
            raise ValueError(
                f"Expected {len(target_layers)} init layers, got {len(layer_indices)} for {repo_id}"
            )

        target_num_heads = self.blocks[0].self_attn.num_heads
        target_num_kv_heads = self.blocks[0].self_attn.num_kv_heads
        target_head_dim = self.blocks[0].self_attn.head_dim
        target_hidden = self.hidden_size

        copied_embeddings = False
        if source.embed_tokens.weight.shape[0] == self.token_embedding.weight.shape[0]:
            _copy_parameter_(
                self.token_embedding.weight,
                _resample_matrix(
                    source.embed_tokens.weight,
                    (self.token_embedding.weight.shape[0], self.token_embedding.weight.shape[1]),
                ),
            )
            copied_embeddings = True

        for target_block, source_index in zip(target_layers, layer_indices):
            source_layer = source_layers[source_index]
            _copy_parameter_(
                target_block.self_norm.weight,
                _resample_vector(source_layer.input_layernorm.weight, target_hidden),
            )
            _copy_parameter_(
                target_block.ffn_norm.weight,
                _resample_vector(source_layer.post_attention_layernorm.weight, target_hidden),
            )

            _copy_parameter_(
                target_block.self_attn.q_proj.weight,
                _resample_attention_weight(
                    source_layer.self_attn.q_proj.weight,
                    source_heads=source_cfg.num_attention_heads,
                    target_heads=target_num_heads,
                    head_dim=target_head_dim,
                    target_input_dim=target_hidden,
                ),
            )
            _copy_parameter_(
                target_block.self_attn.k_proj.weight,
                _resample_attention_weight(
                    source_layer.self_attn.k_proj.weight,
                    source_heads=source_cfg.num_key_value_heads,
                    target_heads=target_num_kv_heads,
                    head_dim=target_head_dim,
                    target_input_dim=target_hidden,
                ),
            )
            _copy_parameter_(
                target_block.self_attn.v_proj.weight,
                _resample_attention_weight(
                    source_layer.self_attn.v_proj.weight,
                    source_heads=source_cfg.num_key_value_heads,
                    target_heads=target_num_kv_heads,
                    head_dim=target_head_dim,
                    target_input_dim=target_hidden,
                ),
            )
            _copy_parameter_(
                target_block.self_attn.o_proj.weight,
                _resample_matrix(
                    source_layer.self_attn.o_proj.weight,
                    target_block.self_attn.o_proj.weight.shape,
                ),
            )
            _copy_parameter_(
                target_block.ffn.gate_proj.weight,
                _resample_matrix(
                    source_layer.mlp.gate_proj.weight,
                    target_block.ffn.gate_proj.weight.shape,
                ),
            )
            _copy_parameter_(
                target_block.ffn.up_proj.weight,
                _resample_matrix(
                    source_layer.mlp.up_proj.weight,
                    target_block.ffn.up_proj.weight.shape,
                ),
            )
            _copy_parameter_(
                target_block.ffn.down_proj.weight,
                _resample_matrix(
                    source_layer.mlp.down_proj.weight,
                    target_block.ffn.down_proj.weight.shape,
                ),
            )

        _copy_parameter_(self.norm.weight, _resample_vector(source.norm.weight, target_hidden))

        self.pretrained_report = {
            "initialized": True,
            "repo_id": repo_id,
            "layer_indices": list(layer_indices),
            "copied_embeddings": copied_embeddings,
            "source_hidden_size": int(source_cfg.hidden_size),
            "target_hidden_size": int(target_hidden),
            "source_num_layers": int(source_cfg.num_hidden_layers),
            "target_num_layers": len(target_layers),
        }
        return self.pretrained_report

    def _project_vocab(self, hidden_states: Tensor) -> Tensor:
        if self.tie_embeddings:
            return F.linear(hidden_states, self.token_embedding.weight)
        return self.output_projection(hidden_states)

    def forward(self, input_ids: Tensor, memory: Tensor) -> tuple[Tensor, tuple[Tensor, ...] | None]:
        B, S = input_ids.shape
        x = self.token_embedding(input_ids)
        x = self.dropout(x)
        causal_mask = make_causal_mask(S, input_ids.device)
        rope = self.rope_cache.to(x.device)
        for block in self.blocks:
            x = block(x, memory, rope_cache=rope, causal_mask=causal_mask)
        x = self.norm(x)
        decoder_logits = self._project_vocab(x)

        mtp_logits: tuple[Tensor, ...] | None = None
        if self.mtp_enabled:
            future = self.mtp_dropout(self.mtp_norm(x))
            mtp_logits = tuple(self._project_vocab(head(future)) for head in self.mtp_heads)
        return decoder_logits, mtp_logits


# ---------------------------------------------------------------------------
# Full Model
# ---------------------------------------------------------------------------

class LeopardiS0(nn.Module):
    def __init__(self, config: LeopardiS0Config) -> None:
        super().__init__()
        self.config = config
        self.canonicalizer = PageCanonicalizer(
            line_density_pool=config.page_canonicalizer.line_density_pool,
            contrast_eps=config.page_canonicalizer.contrast_eps,
        )
        self.vision_encoder = VisionEncoder(config)
        self.layout_side_encoder = LayoutSideMapEncoder(config)
        self.latent_bottleneck = StructuralLatentBottleneck(config)
        self.planner = BlockPlanner(config)
        self.writer = WriterDecoder(config)

        hidden = config.hidden_size
        self.rotation_head = nn.Linear(hidden, config.auxiliary_heads.rotation_classes)
        self.handwriting_head = nn.Linear(hidden, config.auxiliary_heads.handwriting_classes)
        self.formula_token_head = nn.Linear(hidden, 1)
        self.table_block_head = nn.Linear(hidden, 1)
        self.table_span_head = nn.Linear(hidden, config.auxiliary_heads.table_span_dims)
        self.pretrained_state: dict[str, Any] = {
            "vision_encoder_loaded": False,
            "writer_initialized": False,
        }

    @classmethod
    def from_yaml(
        cls,
        path: str,
        load_pretrained: bool = False,
        *,
        local_files_only: bool = False,
    ) -> "LeopardiS0":
        model = cls(LeopardiS0Config.from_yaml(path))
        if load_pretrained:
            model.load_pretrained_components(local_files_only=local_files_only)
        return model

    def load_pretrained_components(self, *, local_files_only: bool = False) -> dict[str, Any]:
        self.vision_encoder.load_pretrained_vision_encoder(local_files_only=local_files_only)
        writer_report = self.writer.initialize_from_pretrained(
            repo_id=self.config.writer_decoder.pretrained_init,
            layer_indices=self.config.writer_decoder.init_layer_indices,
            local_files_only=local_files_only,
        )
        self.pretrained_state = {
            "vision_encoder_loaded": self.vision_encoder.pretrained_loaded,
            "writer_initialized": bool(writer_report.get("initialized", False)),
            "writer": writer_report,
        }
        return self.pretrained_state

    def forward(
        self,
        image: Tensor,
        decoder_input_ids: Tensor,
        visual_mode: str | int | None = None,
    ) -> LeopardiS0Output:
        canonicalized = self.canonicalizer(image)
        visual_tokens, page_summary = self.vision_encoder(canonicalized.image)
        layout_tokens, layout_summary = self.layout_side_encoder(canonicalized)
        context = torch.cat((visual_tokens, layout_tokens), dim=1)
        structural_latents = self.latent_bottleneck(context)
        planner_outputs = self.planner(structural_latents)
        memory = torch.cat((structural_latents, layout_tokens, planner_outputs.states), dim=1)
        decoder_logits, mtp_logits = self.writer(decoder_input_ids, memory)
        combined_summary = 0.5 * (page_summary + layout_summary)

        auxiliary = AuxiliaryOutputs(
            rotation_logits=self.rotation_head(combined_summary),
            handwriting_logits=self.handwriting_head(combined_summary),
            formula_token_logits=self.formula_token_head(visual_tokens).squeeze(-1),
            table_block_logits=self.table_block_head(planner_outputs.states).squeeze(-1),
            table_span_logits=self.table_span_head(planner_outputs.states),
        )
        return LeopardiS0Output(
            canonicalized=canonicalized,
            visual_tokens=visual_tokens,
            layout_tokens=layout_tokens,
            structural_latents=structural_latents,
            planner=planner_outputs,
            decoder_logits=decoder_logits,
            mtp_logits=mtp_logits,
            auxiliary=auxiliary,
        )

    def num_parameters(self, trainable_only: bool = True) -> int:
        if trainable_only:
            return sum(p.numel() for p in self.parameters() if p.requires_grad)
        return sum(p.numel() for p in self.parameters())

    def summary(self) -> dict[str, Any]:
        total = self.num_parameters(trainable_only=False)
        trainable = self.num_parameters(trainable_only=True)
        pending_pretrained_m = max(0.0, float(self.config.target_params_m) - (total / 1_000_000))
        return {
            "family": self.config.family,
            "target_params_m": self.config.target_params_m,
            "total_params_m": round(total / 1_000_000, 2),
            "pending_pretrained_params_m": round(pending_pretrained_m, 2),
            "trainable_params_m": round(trainable / 1_000_000, 2),
            "hidden_size": self.config.hidden_size,
            "vision_encoder": self.config.vision_encoder.pretrained_model or "custom",
            "vision_encoder_loaded": self.pretrained_state.get("vision_encoder_loaded", False),
            "layout_tokens": self.config.layout_side_encoder.pool_grid[0] * self.config.layout_side_encoder.pool_grid[1],
            "num_latents": self.config.latent_bottleneck.num_latents,
            "num_planner_blocks": self.config.planner.num_blocks,
            "decoder_layers": self.config.writer_decoder.num_layers,
            "decoder_pretrained_init": self.config.writer_decoder.pretrained_init or "none",
            "writer_initialized": self.pretrained_state.get("writer_initialized", False),
            "mtp_horizon": self.config.multi_token_prediction.horizon if self.config.multi_token_prediction.enabled else 0,
            "vocab_size": self.config.writer_decoder.vocab_size,
            "max_seq_len": self.config.writer_decoder.max_seq_len,
        }
