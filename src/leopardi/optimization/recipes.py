from __future__ import annotations

from dataclasses import asdict

from leopardi.optimization.config import (
    CalibrationConfig,
    OptimizationGoalConfig,
    OptimizationRuntimeConfig,
    OptimizationStageConfig,
    OptimizationVariantConfig,
)


def optimization_stage_recipe(stage: str) -> OptimizationStageConfig:
    runtime = OptimizationRuntimeConfig()
    goal = OptimizationGoalConfig()
    calibration = CalibrationConfig()

    recipes = {
        "o0_reference_export": OptimizationStageConfig(
            stage="o0_reference_export",
            base_stage="f3_rlvr",
            calibration=calibration,
            runtime=runtime,
            goal=goal,
            variants=(
                OptimizationVariantConfig(
                    variant_id="bf16_reference",
                    method="reference_export",
                    export_format="hf",
                    weight_dtype="bf16",
                    activation_dtype="bf16",
                    kv_cache_dtype="bf16",
                    runtime_targets=("hf", "vllm", "sglang"),
                    requires_calibration=False,
                    expected_latency_gain=0.0,
                    expected_memory_gain=0.0,
                    quality_risk="minimal",
                    notes=("canonical reference artifact for all later optimization comparisons",),
                ),
            ),
        ),
        "o1_torchao_portable": OptimizationStageConfig(
            stage="o1_torchao_portable",
            base_stage="f1_specialist_sft",
            calibration=calibration,
            runtime=runtime,
            goal=goal,
            variants=(
                OptimizationVariantConfig(
                    variant_id="torchao_int4_weight_only",
                    method="torchao_ptq",
                    export_format="hf_torchao",
                    weight_dtype="int4",
                    activation_dtype="bf16",
                    kv_cache_dtype="bf16",
                    runtime_targets=("hf", "vllm"),
                    requires_calibration=False,
                    expected_latency_gain=0.18,
                    expected_memory_gain=0.42,
                    quality_risk="medium",
                    notes=("portable low-bit candidate for local and vLLM-backed serving",),
                ),
                OptimizationVariantConfig(
                    variant_id="torchao_fp8_dynamic",
                    method="torchao_ptq",
                    export_format="hf_torchao",
                    weight_dtype="fp8",
                    activation_dtype="fp8_dynamic",
                    kv_cache_dtype="bf16",
                    runtime_targets=("hf", "vllm"),
                    requires_calibration=False,
                    expected_latency_gain=0.12,
                    expected_memory_gain=0.25,
                    quality_risk="low",
                    notes=("quality-preserving low-risk serving candidate",),
                ),
            ),
        ),
        "o2_vllm_compressed": OptimizationStageConfig(
            stage="o2_vllm_compressed",
            base_stage="f3_rlvr",
            calibration=CalibrationConfig(
                bundle_id="o_calibration_docmix_v1",
                max_samples=768,
                max_seq_len=2048,
                page_mix=("medium", "hard"),
                include_tables=True,
                include_formulas=True,
                include_handwriting=True,
                include_charts=True,
            ),
            runtime=runtime,
            goal=goal,
            variants=(
                OptimizationVariantConfig(
                    variant_id="llmcompressor_fp8_dynamic",
                    method="llmcompressor_fp8",
                    export_format="compressed_tensors",
                    weight_dtype="fp8_block",
                    activation_dtype="fp8_dynamic",
                    kv_cache_dtype="bf16",
                    runtime_targets=("vllm",),
                    requires_calibration=False,
                    expected_latency_gain=0.2,
                    expected_memory_gain=0.28,
                    quality_risk="low",
                    notes=("best low-risk vLLM-specific acceleration path",),
                ),
                OptimizationVariantConfig(
                    variant_id="llmcompressor_w4a16_awq",
                    method="llmcompressor_awq",
                    export_format="compressed_tensors",
                    weight_dtype="int4",
                    activation_dtype="bf16",
                    kv_cache_dtype="bf16",
                    runtime_targets=("vllm",),
                    requires_calibration=True,
                    calibration_samples=768,
                    expected_latency_gain=0.3,
                    expected_memory_gain=0.5,
                    quality_risk="medium",
                    notes=("aggressive deployment candidate with calibration",),
                ),
            ),
        ),
        "o3_runtime_kv": OptimizationStageConfig(
            stage="o3_runtime_kv",
            base_stage="o2_vllm_compressed",
            calibration=CalibrationConfig(
                bundle_id="o_calibration_docmix_v1",
                max_samples=256,
                max_seq_len=2048,
                page_mix=("easy", "medium", "hard"),
                include_tables=True,
                include_formulas=True,
                include_handwriting=False,
                include_charts=False,
            ),
            runtime=runtime,
            goal=goal,
            variants=(
                OptimizationVariantConfig(
                    variant_id="vllm_fp8_kv",
                    method="runtime_kv_quant",
                    export_format="runtime_only",
                    weight_dtype="inherit",
                    activation_dtype="inherit",
                    kv_cache_dtype="fp8",
                    runtime_targets=("vllm",),
                    requires_calibration=False,
                    expected_latency_gain=0.08,
                    expected_memory_gain=0.15,
                    quality_risk="low",
                    notes=("runtime-only variant layered on a promoted checkpoint",),
                ),
                OptimizationVariantConfig(
                    variant_id="sglang_fp8_kv",
                    method="runtime_kv_quant",
                    export_format="runtime_only",
                    weight_dtype="inherit",
                    activation_dtype="inherit",
                    kv_cache_dtype="fp8",
                    runtime_targets=("sglang",),
                    requires_calibration=False,
                    expected_latency_gain=0.08,
                    expected_memory_gain=0.15,
                    quality_risk="low",
                    notes=("runtime-only SGLang KV candidate once serving path is stable",),
                ),
            ),
        ),
        "o4_qat_export": OptimizationStageConfig(
            stage="o4_qat_export",
            base_stage="f1_specialist_sft",
            calibration=calibration,
            runtime=runtime,
            goal=OptimizationGoalConfig(
                target_frontier="quality_speed_memory",
                quality_floor=0.94,
                markdown_validity_floor=0.995,
                latex_validity_floor=0.99,
                max_relative_quality_drop=0.01,
                min_latency_gain=0.1,
                max_memory_gb=20.0,
            ),
            variants=(
                OptimizationVariantConfig(
                    variant_id="qat_int4_export",
                    method="torchao_qat_convert",
                    export_format="hf_torchao",
                    weight_dtype="int4",
                    activation_dtype="int8_dynamic",
                    kv_cache_dtype="bf16",
                    runtime_targets=("hf", "vllm"),
                    requires_calibration=False,
                    expected_latency_gain=0.22,
                    expected_memory_gain=0.45,
                    quality_risk="low",
                    notes=("only valid after a QAT-aware finetune branch",),
                ),
            ),
        ),
    }
    if stage not in recipes:
        raise KeyError(f"Unknown optimization stage recipe: {stage}")
    return recipes[stage]


def optimization_stage_recipe_dict(stage: str) -> dict[str, object]:
    return asdict(optimization_stage_recipe(stage))
