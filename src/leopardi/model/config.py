from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml


@dataclass(slots=True)
class PageCanonicalizerConfig:
    line_density_pool: int = 16
    contrast_eps: float = 1e-5


@dataclass(slots=True)
class VisionEncoderConfig:
    pretrained_model: str = "google/siglip2-base-patch16-naflex"
    output_dim: int = 768
    freeze_layers: int = 8
    pixel_shuffle_factor: int = 2
    projection_dim: int = 576


@dataclass(slots=True)
class LayoutSideEncoderConfig:
    stem_dim: int = 64
    hidden_size: int = 576
    pool_grid: tuple[int, int] = (3, 4)
    dropout: float = 0.0


@dataclass(slots=True)
class LatentBottleneckConfig:
    hidden_size: int = 576
    num_latents: int = 192
    num_layers: int = 3
    num_heads: int = 9
    num_kv_heads: int = 3
    mlp_ratio: float = 2.6666666667
    dropout: float = 0.1


@dataclass(slots=True)
class PlannerConfig:
    hidden_size: int = 576
    num_layers: int = 3
    num_heads: int = 9
    num_kv_heads: int = 3
    mlp_ratio: float = 2.6666666667
    num_blocks: int = 64
    num_length_buckets: int = 8
    block_types: tuple[str, ...] = (
        "heading",
        "paragraph",
        "list",
        "table",
        "figure",
        "figure_caption",
        "equation",
        "page_header",
        "page_footer",
        "marginalia",
    )
    specialist_hints: tuple[str, ...] = ("default", "math", "table", "handwriting", "chart")
    dropout: float = 0.1


@dataclass(slots=True)
class WriterDecoderConfig:
    vocab_size: int = 40_960
    hidden_size: int = 576
    num_layers: int = 12
    num_heads: int = 9
    num_kv_heads: int = 3
    mlp_ratio: float = 2.6666666667
    max_seq_len: int = 4_096
    rope_theta: float = 1_000_000.0
    dropout: float = 0.1
    tie_embeddings: bool = True
    pretrained_init: str = ""
    init_layer_indices: list[int] | None = None


@dataclass(slots=True)
class MultiTokenPredictionConfig:
    enabled: bool = True
    horizon: int = 2
    dropout: float = 0.05


@dataclass(slots=True)
class AuxiliaryHeadsConfig:
    rotation_classes: int = 4
    handwriting_classes: int = 4
    table_span_dims: int = 4


@dataclass(slots=True)
class LeopardiS0Config:
    family: str = "leopardi_s0"
    target_params_m: int = 200
    hidden_size: int = 576
    page_canonicalizer: PageCanonicalizerConfig = field(default_factory=PageCanonicalizerConfig)
    vision_encoder: VisionEncoderConfig = field(default_factory=VisionEncoderConfig)
    layout_side_encoder: LayoutSideEncoderConfig = field(default_factory=LayoutSideEncoderConfig)
    latent_bottleneck: LatentBottleneckConfig = field(default_factory=LatentBottleneckConfig)
    planner: PlannerConfig = field(default_factory=PlannerConfig)
    writer_decoder: WriterDecoderConfig = field(default_factory=WriterDecoderConfig)
    multi_token_prediction: MultiTokenPredictionConfig = field(default_factory=MultiTokenPredictionConfig)
    auxiliary_heads: AuxiliaryHeadsConfig = field(default_factory=AuxiliaryHeadsConfig)

    def __post_init__(self) -> None:
        expected = (
            self.vision_encoder.projection_dim,
            self.layout_side_encoder.hidden_size,
            self.latent_bottleneck.hidden_size,
            self.planner.hidden_size,
            self.writer_decoder.hidden_size,
        )
        if len(set(expected)) != 1:
            raise ValueError(
                f"Internal hidden sizes must match: {expected}. "
                "Vision projection_dim, layout, bottleneck, planner, and "
                "decoder hidden_size must all be equal."
            )
        self.hidden_size = expected[0]
        attention_specs = (
            ("latent_bottleneck", self.latent_bottleneck.num_heads, self.latent_bottleneck.num_kv_heads),
            ("planner", self.planner.num_heads, self.planner.num_kv_heads),
            ("writer_decoder", self.writer_decoder.num_heads, self.writer_decoder.num_kv_heads),
        )
        for name, heads, kv_heads in attention_specs:
            if self.hidden_size % heads != 0:
                raise ValueError(f"{name}.num_heads={heads} must divide hidden_size={self.hidden_size}")
            if heads % kv_heads != 0:
                raise ValueError(f"{name}.num_heads={heads} must be divisible by num_kv_heads={kv_heads}")

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "LeopardiS0Config":
        model = payload.get("model", payload)

        writer_kwargs = dict(model.get("writer_decoder", {}))
        if "init_layer_indices" in writer_kwargs and writer_kwargs["init_layer_indices"] is not None:
            writer_kwargs["init_layer_indices"] = list(writer_kwargs["init_layer_indices"])

        return cls(
            family=model.get("family", "leopardi_s0"),
            target_params_m=model.get("target_params_m", 200),
            hidden_size=model.get("hidden_size", 576),
            page_canonicalizer=PageCanonicalizerConfig(
                **model.get("page_canonicalizer", {})
            ),
            vision_encoder=VisionEncoderConfig(**model.get("vision_encoder", {})),
            layout_side_encoder=LayoutSideEncoderConfig(**model.get("layout_side_encoder", {})),
            latent_bottleneck=LatentBottleneckConfig(**model.get("latent_bottleneck", {})),
            planner=PlannerConfig(**model.get("planner", {})),
            writer_decoder=WriterDecoderConfig(**writer_kwargs),
            multi_token_prediction=MultiTokenPredictionConfig(**model.get("multi_token_prediction", {})),
            auxiliary_heads=AuxiliaryHeadsConfig(**model.get("auxiliary_heads", {})),
        )

    @classmethod
    def from_yaml(cls, path: str | Path) -> "LeopardiS0Config":
        with Path(path).open("r", encoding="utf-8") as handle:
            payload = yaml.safe_load(handle)
        return cls.from_dict(payload)

    @classmethod
    def tiny(cls) -> "LeopardiS0Config":
        return cls(
            hidden_size=128,
            target_params_m=8,
            vision_encoder=VisionEncoderConfig(
                pretrained_model="",
                output_dim=128,
                freeze_layers=0,
                pixel_shuffle_factor=1,
                projection_dim=128,
            ),
            layout_side_encoder=LayoutSideEncoderConfig(
                stem_dim=24,
                hidden_size=128,
                pool_grid=(2, 2),
            ),
            latent_bottleneck=LatentBottleneckConfig(
                hidden_size=128,
                num_latents=32,
                num_layers=2,
                num_heads=4,
                num_kv_heads=2,
                mlp_ratio=2.67,
            ),
            planner=PlannerConfig(
                hidden_size=128,
                num_layers=1,
                num_heads=4,
                num_kv_heads=2,
                mlp_ratio=2.67,
                num_blocks=12,
                num_length_buckets=4,
            ),
            writer_decoder=WriterDecoderConfig(
                vocab_size=512,
                hidden_size=128,
                num_layers=2,
                num_heads=4,
                num_kv_heads=2,
                mlp_ratio=2.67,
                max_seq_len=128,
                rope_theta=10_000.0,
            ),
            multi_token_prediction=MultiTokenPredictionConfig(
                enabled=True,
                horizon=1,
            ),
            auxiliary_heads=AuxiliaryHeadsConfig(
                rotation_classes=4,
                handwriting_classes=3,
                table_span_dims=4,
            ),
        )
