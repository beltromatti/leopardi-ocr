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
class VisualTokenizerConfig:
    stem_dim: int = 96
    stage_dims: tuple[int, int, int, int] = (96, 160, 224, 320)
    stage_depths: tuple[int, int, int, int] = (2, 2, 4, 2)
    hidden_size: int = 384
    dropout: float = 0.0
    pool_layouts: dict[str, list[list[int]]] = field(
        default_factory=lambda: {
            "fast": [[4, 4], [6, 6]],
            "standard": [[6, 6], [8, 8]],
            "hard": [[8, 8], [10, 10]],
        }
    )
    stage_indices: tuple[int, int] = (2, 3)


@dataclass(slots=True)
class LayoutSideEncoderConfig:
    stem_dim: int = 48
    hidden_size: int = 384
    pool_grid: tuple[int, int] = (3, 4)
    dropout: float = 0.0


@dataclass(slots=True)
class LatentBottleneckConfig:
    hidden_size: int = 384
    num_latents: int = 192
    num_layers: int = 4
    num_heads: int = 6
    mlp_ratio: float = 4.0
    dropout: float = 0.1


@dataclass(slots=True)
class PlannerConfig:
    hidden_size: int = 384
    num_layers: int = 3
    num_heads: int = 6
    num_blocks: int = 48
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
    hidden_size: int = 384
    num_layers: int = 8
    num_heads: int = 6
    mlp_ratio: float = 4.0
    max_seq_len: int = 2_048
    dropout: float = 0.1
    tie_embeddings: bool = True


@dataclass(slots=True)
class MultiTokenPredictionConfig:
    enabled: bool = True
    horizon: int = 2
    dropout: float = 0.0


@dataclass(slots=True)
class AuxiliaryHeadsConfig:
    rotation_classes: int = 4
    handwriting_classes: int = 4
    table_span_dims: int = 4


@dataclass(slots=True)
class LeopardiS0Config:
    family: str = "leopardi_s0"
    target_params_m: int = 100
    hidden_size: int = 384
    page_canonicalizer: PageCanonicalizerConfig = field(default_factory=PageCanonicalizerConfig)
    visual_tokenizer: VisualTokenizerConfig = field(default_factory=VisualTokenizerConfig)
    layout_side_encoder: LayoutSideEncoderConfig = field(default_factory=LayoutSideEncoderConfig)
    latent_bottleneck: LatentBottleneckConfig = field(default_factory=LatentBottleneckConfig)
    planner: PlannerConfig = field(default_factory=PlannerConfig)
    writer_decoder: WriterDecoderConfig = field(default_factory=WriterDecoderConfig)
    multi_token_prediction: MultiTokenPredictionConfig = field(default_factory=MultiTokenPredictionConfig)
    auxiliary_heads: AuxiliaryHeadsConfig = field(default_factory=AuxiliaryHeadsConfig)

    def __post_init__(self) -> None:
        expected = (
            self.visual_tokenizer.hidden_size,
            self.layout_side_encoder.hidden_size,
            self.latent_bottleneck.hidden_size,
            self.planner.hidden_size,
            self.writer_decoder.hidden_size,
        )
        if len(set(expected)) != 1:
            raise ValueError("All hidden sizes must match for Leopardi-S0.")
        self.hidden_size = expected[0]

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "LeopardiS0Config":
        model = payload.get("model", payload)
        return cls(
            family=model.get("family", "leopardi_s0"),
            target_params_m=model.get("target_params_m", 100),
            hidden_size=model.get("hidden_size", 384),
            page_canonicalizer=PageCanonicalizerConfig(
                **model.get("page_canonicalizer", {})
            ),
            visual_tokenizer=VisualTokenizerConfig(**model.get("visual_tokenizer", {})),
            layout_side_encoder=LayoutSideEncoderConfig(**model.get("layout_side_encoder", {})),
            latent_bottleneck=LatentBottleneckConfig(**model.get("latent_bottleneck", {})),
            planner=PlannerConfig(**model.get("planner", {})),
            writer_decoder=WriterDecoderConfig(**model.get("writer_decoder", {})),
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
            visual_tokenizer=VisualTokenizerConfig(
                stem_dim=32,
                stage_dims=(32, 48, 64, 96),
                stage_depths=(1, 1, 1, 1),
                hidden_size=128,
                pool_layouts={"fast": [[2, 2], [3, 3]], "standard": [[3, 3], [4, 4]], "hard": [[4, 4], [5, 5]]},
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
            ),
            planner=PlannerConfig(
                hidden_size=128,
                num_layers=2,
                num_heads=4,
                num_blocks=12,
                num_length_buckets=4,
            ),
            writer_decoder=WriterDecoderConfig(
                vocab_size=512,
                hidden_size=128,
                num_layers=2,
                num_heads=4,
                max_seq_len=128,
            ),
            multi_token_prediction=MultiTokenPredictionConfig(
                enabled=True,
                horizon=2,
            ),
            auxiliary_heads=AuxiliaryHeadsConfig(
                rotation_classes=4,
                handwriting_classes=3,
                table_span_dims=4,
            ),
        )
