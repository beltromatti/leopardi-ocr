from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path

from leopardi.optimization.config import OptimizationStageConfig, OptimizationVariantConfig


@dataclass(slots=True)
class OptimizationArtifactCard:
    experiment_id: str
    stage: str
    base_stage: str
    base_checkpoint_tag: str
    base_checkpoint_uri: str
    variant_id: str
    method: str
    export_format: str
    runtime_targets: tuple[str, ...]
    structured_output_backend: str
    weight_dtype: str
    activation_dtype: str
    kv_cache_dtype: str
    requires_calibration: bool
    calibration_bundle_id: str | None
    calibration_samples: int
    artifact_dir: str
    persistent_artifact_uri: str
    promotion_gate: str
    target_frontier: str
    quality_floor: float
    markdown_validity_floor: float
    latex_validity_floor: float
    max_relative_quality_drop: float
    min_latency_gain: float
    max_memory_gb: float
    expected_latency_gain: float
    expected_memory_gain: float
    quality_risk: str
    notes: tuple[str, ...]


def build_artifact_card(
    *,
    experiment_id: str,
    stage: OptimizationStageConfig,
    variant: OptimizationVariantConfig,
    artifact_dir: str | Path,
    persistent_artifact_uri: str,
    base_checkpoint_uri: str,
    promotion_gate: str = "release_gate_v1",
) -> OptimizationArtifactCard:
    calibration_samples = variant.calibration_samples or stage.calibration.max_samples
    return OptimizationArtifactCard(
        experiment_id=experiment_id,
        stage=stage.stage,
        base_stage=stage.base_stage,
        base_checkpoint_tag=stage.base_checkpoint_tag,
        base_checkpoint_uri=base_checkpoint_uri,
        variant_id=variant.variant_id,
        method=variant.method,
        export_format=variant.export_format,
        runtime_targets=variant.runtime_targets,
        structured_output_backend=variant.structured_output_backend,
        weight_dtype=variant.weight_dtype,
        activation_dtype=variant.activation_dtype,
        kv_cache_dtype=variant.kv_cache_dtype,
        requires_calibration=variant.requires_calibration,
        calibration_bundle_id=stage.calibration.bundle_id if variant.requires_calibration else None,
        calibration_samples=calibration_samples if variant.requires_calibration else 0,
        artifact_dir=str(artifact_dir),
        persistent_artifact_uri=persistent_artifact_uri,
        promotion_gate=promotion_gate,
        target_frontier=stage.goal.target_frontier,
        quality_floor=stage.goal.quality_floor,
        markdown_validity_floor=stage.goal.markdown_validity_floor,
        latex_validity_floor=stage.goal.latex_validity_floor,
        max_relative_quality_drop=stage.goal.max_relative_quality_drop,
        min_latency_gain=stage.goal.min_latency_gain,
        max_memory_gb=stage.goal.max_memory_gb,
        expected_latency_gain=variant.expected_latency_gain,
        expected_memory_gain=variant.expected_memory_gain,
        quality_risk=variant.quality_risk,
        notes=variant.notes,
    )


def artifact_card_dict(card: OptimizationArtifactCard) -> dict[str, object]:
    return asdict(card)
