from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml


@dataclass(slots=True)
class CalibrationConfig:
    bundle_id: str = "o_calibration_docmix_v1"
    max_samples: int = 512
    max_seq_len: int = 2048
    page_mix: tuple[str, ...] = ("easy", "medium", "hard")
    include_tables: bool = True
    include_formulas: bool = True
    include_handwriting: bool = True
    include_charts: bool = True


@dataclass(slots=True)
class OptimizationRuntimeConfig:
    hardware_tag: str = "rtx5090"
    precision: str = "bf16"
    primary_runtime: str = "vllm"
    alternate_runtime: str = "sglang"
    calibration_workers: int = 4
    export_workers: int = 2
    max_calibration_samples: int = 768
    log_every: int = 10
    eval_every: int = 1
    save_every: int = 1


@dataclass(slots=True)
class OptimizationGoalConfig:
    target_frontier: str = "quality_speed_memory"
    quality_floor: float = 0.93
    markdown_validity_floor: float = 0.995
    latex_validity_floor: float = 0.99
    max_relative_quality_drop: float = 0.015
    min_latency_gain: float = 0.15
    max_memory_gb: float = 20.0


@dataclass(slots=True)
class OptimizationVariantConfig:
    variant_id: str
    method: str
    export_format: str = "hf"
    weight_dtype: str = "bf16"
    activation_dtype: str = "bf16"
    kv_cache_dtype: str = "bf16"
    runtime_targets: tuple[str, ...] = ("hf",)
    structured_output_backend: str = "auto"
    requires_calibration: bool = False
    calibration_samples: int | None = None
    expected_latency_gain: float = 0.0
    expected_memory_gain: float = 0.0
    quality_risk: str = "low"
    notes: tuple[str, ...] = ()


@dataclass(slots=True)
class OptimizationStageConfig:
    stage: str
    track: str = "s0-runtime"
    base_stage: str = "f3_rlvr"
    base_checkpoint_tag: str = "candidate"
    calibration: CalibrationConfig = field(default_factory=CalibrationConfig)
    runtime: OptimizationRuntimeConfig = field(default_factory=OptimizationRuntimeConfig)
    goal: OptimizationGoalConfig = field(default_factory=OptimizationGoalConfig)
    variants: tuple[OptimizationVariantConfig, ...] = ()

    @classmethod
    def from_dict(
        cls,
        payload: dict[str, Any],
        runtime_payload: dict[str, Any] | None = None,
    ) -> "OptimizationStageConfig":
        runtime_root = (runtime_payload or {}).get("runtime", runtime_payload or {})
        calibration_payload = payload.get("calibration", {})
        goal_payload = payload.get("goal", {})
        variants = tuple(
            OptimizationVariantConfig(
                variant_id=variant["variant_id"],
                method=variant["method"],
                export_format=variant.get("export_format", "hf"),
                weight_dtype=variant.get("weight_dtype", "bf16"),
                activation_dtype=variant.get("activation_dtype", "bf16"),
                kv_cache_dtype=variant.get("kv_cache_dtype", "bf16"),
                runtime_targets=tuple(variant.get("runtime_targets", ("hf",))),
                structured_output_backend=variant.get("structured_output_backend", "auto"),
                requires_calibration=variant.get("requires_calibration", False),
                calibration_samples=variant.get("calibration_samples"),
                expected_latency_gain=variant.get("expected_latency_gain", 0.0),
                expected_memory_gain=variant.get("expected_memory_gain", 0.0),
                quality_risk=variant.get("quality_risk", "low"),
                notes=tuple(variant.get("notes", ())),
            )
            for variant in payload.get("variants", [])
        )
        return cls(
            stage=payload["stage"],
            track=payload.get("track", "s0-runtime"),
            base_stage=payload.get("base_stage", "f3_rlvr"),
            base_checkpoint_tag=payload.get("base_checkpoint_tag", "candidate"),
            calibration=CalibrationConfig(
                bundle_id=calibration_payload.get("bundle_id", "o_calibration_docmix_v1"),
                max_samples=calibration_payload.get("max_samples", 512),
                max_seq_len=calibration_payload.get("max_seq_len", 2048),
                page_mix=tuple(calibration_payload.get("page_mix", ("easy", "medium", "hard"))),
                include_tables=calibration_payload.get("include_tables", True),
                include_formulas=calibration_payload.get("include_formulas", True),
                include_handwriting=calibration_payload.get("include_handwriting", True),
                include_charts=calibration_payload.get("include_charts", True),
            ),
            runtime=OptimizationRuntimeConfig(
                hardware_tag=runtime_root.get("hardware_tag", "rtx5090"),
                precision=runtime_root.get("precision", "bf16"),
                primary_runtime=runtime_root.get("primary_runtime", "vllm"),
                alternate_runtime=runtime_root.get("alternate_runtime", "sglang"),
                calibration_workers=runtime_root.get("calibration_workers", 4),
                export_workers=runtime_root.get("export_workers", 2),
                max_calibration_samples=runtime_root.get("max_calibration_samples", 768),
                log_every=runtime_root.get("log_every", 10),
                eval_every=runtime_root.get("eval_every", 1),
                save_every=runtime_root.get("save_every", 1),
            ),
            goal=OptimizationGoalConfig(
                target_frontier=goal_payload.get("target_frontier", "quality_speed_memory"),
                quality_floor=goal_payload.get("quality_floor", 0.93),
                markdown_validity_floor=goal_payload.get("markdown_validity_floor", 0.995),
                latex_validity_floor=goal_payload.get("latex_validity_floor", 0.99),
                max_relative_quality_drop=goal_payload.get("max_relative_quality_drop", 0.015),
                min_latency_gain=goal_payload.get("min_latency_gain", 0.15),
                max_memory_gb=goal_payload.get("max_memory_gb", 20.0),
            ),
            variants=variants,
        )

    @classmethod
    def from_yaml(
        cls,
        stage_path: str | Path,
        runtime_path: str | Path | None = None,
    ) -> "OptimizationStageConfig":
        with Path(stage_path).open("r", encoding="utf-8") as handle:
            stage_payload = yaml.safe_load(handle)
        runtime_payload: dict[str, Any] | None = None
        if runtime_path is not None:
            with Path(runtime_path).open("r", encoding="utf-8") as handle:
                runtime_payload = yaml.safe_load(handle)
        return cls.from_dict(stage_payload, runtime_payload=runtime_payload)
