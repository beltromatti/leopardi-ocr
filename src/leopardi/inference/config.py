from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml


@dataclass(slots=True)
class RenderConfig:
    target_longest_image_dim: int = 1280
    max_page_pixels: int = 2_200_000
    permit_auto_rotation: bool = True
    permit_crop_refinement: bool = True
    side_map_channels: tuple[str, ...] = ("grayscale", "edges", "contrast")


@dataclass(slots=True)
class RoutingConfig:
    standard_threshold: float = 0.34
    hard_threshold: float = 0.62
    hard_formula_density: float = 0.08
    hard_table_density: float = 0.12
    hard_handwriting_likelihood: float = 0.4
    hard_chart_likelihood: float = 0.35
    hard_long_tiny_text_likelihood: float = 0.45
    hard_photo_distortion_likelihood: float = 0.35
    hard_orientation_uncertainty: float = 0.2
    visual_density_weight: float = 0.2
    block_count_weight: float = 0.1
    formula_density_weight: float = 0.2
    table_density_weight: float = 0.18
    handwriting_weight: float = 0.12
    chart_weight: float = 0.08
    long_tiny_text_weight: float = 0.07
    photo_distortion_weight: float = 0.05


@dataclass(slots=True)
class DecodeModeConfig:
    name: str
    visual_token_budget: int
    max_output_tokens: int
    crop_budget: int
    repair_budget: int
    grammar_mode: str = "light"
    structured_output_backend: str = "auto"
    allow_block_local_repair: bool = True
    specialist_hints: tuple[str, ...] = ()


@dataclass(slots=True)
class ValidationConfig:
    require_balanced_code_fences: bool = True
    require_balanced_math_delimiters: bool = True
    require_table_shape_checks: bool = True
    allow_html_tables: bool = True
    max_error_count_before_hard_fail: int = 2


@dataclass(slots=True)
class AssemblyConfig:
    suppress_repeated_headers: bool = True
    suppress_repeated_footers: bool = True
    min_repeat_pages: int = 2
    emit_page_break_markers: bool = False
    page_break_marker: str = "<!-- page-break -->"


@dataclass(slots=True)
class InferenceRuntimeConfig:
    hardware_tag: str = "rtx5090"
    primary_runtime: str = "vllm"
    alternate_runtime: str = "sglang"
    model_dir_env: str = "LEOPARDI_MODEL_DIR"
    host: str = "127.0.0.1"
    vllm_port: int = 30000
    sglang_port: int = 30001
    gpu_memory_utilization: float = 0.88
    mem_fraction_static: float = 0.82
    max_model_len: int = 8192
    max_num_seqs: int = 16
    tensor_parallel_size: int = 1
    enable_prefix_caching: bool = True
    chunked_prefill: bool = True
    log_every: int = 10
    eval_every: int = 1
    save_every: int = 1


@dataclass(slots=True)
class InferenceStageConfig:
    stage: str
    track: str = "s0-runtime"
    artifact_variant_id: str = "bf16_reference"
    artifact_uri: str = "hf://leopardi-ocr-checkpoints/leo-s0-reference"
    runtime_family: str = "vllm"
    fallback_runtime_family: str = "sglang"
    structured_backend_default: str = "xgrammar"
    render: RenderConfig = field(default_factory=RenderConfig)
    routing: RoutingConfig = field(default_factory=RoutingConfig)
    validation: ValidationConfig = field(default_factory=ValidationConfig)
    assembly: AssemblyConfig = field(default_factory=AssemblyConfig)
    runtime: InferenceRuntimeConfig = field(default_factory=InferenceRuntimeConfig)
    modes: tuple[DecodeModeConfig, ...] = ()

    @classmethod
    def from_dict(
        cls,
        payload: dict[str, Any],
        runtime_payload: dict[str, Any] | None = None,
    ) -> "InferenceStageConfig":
        runtime_root = (runtime_payload or {}).get("runtime", runtime_payload or {})
        render_payload = payload.get("render", {})
        routing_payload = payload.get("routing", {})
        validation_payload = payload.get("validation", {})
        assembly_payload = payload.get("assembly", {})
        modes = tuple(
            DecodeModeConfig(
                name=mode["name"],
                visual_token_budget=mode["visual_token_budget"],
                max_output_tokens=mode["max_output_tokens"],
                crop_budget=mode["crop_budget"],
                repair_budget=mode["repair_budget"],
                grammar_mode=mode.get("grammar_mode", "light"),
                structured_output_backend=mode.get("structured_output_backend", "auto"),
                allow_block_local_repair=mode.get("allow_block_local_repair", True),
                specialist_hints=tuple(mode.get("specialist_hints", ())),
            )
            for mode in payload.get("modes", [])
        )
        return cls(
            stage=payload["stage"],
            track=payload.get("track", "s0-runtime"),
            artifact_variant_id=payload.get("artifact_variant_id", "bf16_reference"),
            artifact_uri=payload.get("artifact_uri", "hf://leopardi-ocr-checkpoints/leo-s0-reference"),
            runtime_family=runtime_root.get("primary_runtime", payload.get("runtime_family", "vllm")),
            fallback_runtime_family=runtime_root.get(
                "alternate_runtime",
                payload.get("fallback_runtime_family", "sglang"),
            ),
            structured_backend_default=payload.get("structured_backend_default", "xgrammar"),
            render=RenderConfig(
                target_longest_image_dim=render_payload.get("target_longest_image_dim", 1280),
                max_page_pixels=render_payload.get("max_page_pixels", 2_200_000),
                permit_auto_rotation=render_payload.get("permit_auto_rotation", True),
                permit_crop_refinement=render_payload.get("permit_crop_refinement", True),
                side_map_channels=tuple(render_payload.get("side_map_channels", ("grayscale", "edges", "contrast"))),
            ),
            routing=RoutingConfig(
                standard_threshold=routing_payload.get("standard_threshold", 0.34),
                hard_threshold=routing_payload.get("hard_threshold", 0.62),
                hard_formula_density=routing_payload.get("hard_formula_density", 0.08),
                hard_table_density=routing_payload.get("hard_table_density", 0.12),
                hard_handwriting_likelihood=routing_payload.get("hard_handwriting_likelihood", 0.4),
                hard_chart_likelihood=routing_payload.get("hard_chart_likelihood", 0.35),
                hard_long_tiny_text_likelihood=routing_payload.get("hard_long_tiny_text_likelihood", 0.45),
                hard_photo_distortion_likelihood=routing_payload.get("hard_photo_distortion_likelihood", 0.35),
                hard_orientation_uncertainty=routing_payload.get("hard_orientation_uncertainty", 0.2),
                visual_density_weight=routing_payload.get("visual_density_weight", 0.2),
                block_count_weight=routing_payload.get("block_count_weight", 0.1),
                formula_density_weight=routing_payload.get("formula_density_weight", 0.2),
                table_density_weight=routing_payload.get("table_density_weight", 0.18),
                handwriting_weight=routing_payload.get("handwriting_weight", 0.12),
                chart_weight=routing_payload.get("chart_weight", 0.08),
                long_tiny_text_weight=routing_payload.get("long_tiny_text_weight", 0.07),
                photo_distortion_weight=routing_payload.get("photo_distortion_weight", 0.05),
            ),
            validation=ValidationConfig(
                require_balanced_code_fences=validation_payload.get("require_balanced_code_fences", True),
                require_balanced_math_delimiters=validation_payload.get("require_balanced_math_delimiters", True),
                require_table_shape_checks=validation_payload.get("require_table_shape_checks", True),
                allow_html_tables=validation_payload.get("allow_html_tables", True),
                max_error_count_before_hard_fail=validation_payload.get("max_error_count_before_hard_fail", 2),
            ),
            assembly=AssemblyConfig(
                suppress_repeated_headers=assembly_payload.get("suppress_repeated_headers", True),
                suppress_repeated_footers=assembly_payload.get("suppress_repeated_footers", True),
                min_repeat_pages=assembly_payload.get("min_repeat_pages", 2),
                emit_page_break_markers=assembly_payload.get("emit_page_break_markers", False),
                page_break_marker=assembly_payload.get("page_break_marker", "<!-- page-break -->"),
            ),
            runtime=InferenceRuntimeConfig(
                hardware_tag=runtime_root.get("hardware_tag", "rtx5090"),
                primary_runtime=runtime_root.get("primary_runtime", payload.get("runtime_family", "vllm")),
                alternate_runtime=runtime_root.get("alternate_runtime", payload.get("fallback_runtime_family", "sglang")),
                model_dir_env=runtime_root.get("model_dir_env", "LEOPARDI_MODEL_DIR"),
                host=runtime_root.get("host", "127.0.0.1"),
                vllm_port=runtime_root.get("vllm_port", 30000),
                sglang_port=runtime_root.get("sglang_port", 30001),
                gpu_memory_utilization=runtime_root.get("gpu_memory_utilization", 0.88),
                mem_fraction_static=runtime_root.get("mem_fraction_static", 0.82),
                max_model_len=runtime_root.get("max_model_len", 8192),
                max_num_seqs=runtime_root.get("max_num_seqs", 16),
                tensor_parallel_size=runtime_root.get("tensor_parallel_size", 1),
                enable_prefix_caching=runtime_root.get("enable_prefix_caching", True),
                chunked_prefill=runtime_root.get("chunked_prefill", True),
                log_every=runtime_root.get("log_every", 10),
                eval_every=runtime_root.get("eval_every", 1),
                save_every=runtime_root.get("save_every", 1),
            ),
            modes=modes,
        )

    @classmethod
    def from_yaml(
        cls,
        stage_path: str | Path,
        runtime_path: str | Path | None = None,
    ) -> "InferenceStageConfig":
        with Path(stage_path).open("r", encoding="utf-8") as handle:
            stage_payload = yaml.safe_load(handle)
        runtime_payload: dict[str, Any] | None = None
        if runtime_path is not None:
            with Path(runtime_path).open("r", encoding="utf-8") as handle:
                runtime_payload = yaml.safe_load(handle)
        return cls.from_dict(stage_payload, runtime_payload=runtime_payload)

    def mode(self, name: str) -> DecodeModeConfig:
        for mode in self.modes:
            if mode.name == name:
                return mode
        raise KeyError(f"Unknown inference mode: {name}")
