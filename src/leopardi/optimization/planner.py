from __future__ import annotations

from dataclasses import asdict

from leopardi.optimization.config import OptimizationStageConfig


def build_variant_plan(stage: OptimizationStageConfig) -> list[dict[str, object]]:
    plan: list[dict[str, object]] = []
    for variant in stage.variants:
        calibration_samples = variant.calibration_samples or stage.calibration.max_samples
        plan.append(
            {
                "variant_id": variant.variant_id,
                "method": variant.method,
                "runtime_targets": list(variant.runtime_targets),
                "export_format": variant.export_format,
                "weight_dtype": variant.weight_dtype,
                "activation_dtype": variant.activation_dtype,
                "kv_cache_dtype": variant.kv_cache_dtype,
                "requires_calibration": variant.requires_calibration,
                "calibration_samples": calibration_samples if variant.requires_calibration else 0,
                "expected_latency_gain": variant.expected_latency_gain,
                "expected_memory_gain": variant.expected_memory_gain,
                "quality_risk": variant.quality_risk,
            }
        )
    return plan


def build_variant_summary(stage: OptimizationStageConfig) -> dict[str, object]:
    return {
        "stage": stage.stage,
        "track": stage.track,
        "base_stage": stage.base_stage,
        "base_checkpoint_tag": stage.base_checkpoint_tag,
        "calibration": asdict(stage.calibration),
        "goal": asdict(stage.goal),
        "runtime": asdict(stage.runtime),
        "variants": build_variant_plan(stage),
    }
