from __future__ import annotations

from dataclasses import asdict

from leopardi.inference.config import (
    AssemblyConfig,
    DecodeModeConfig,
    InferenceRuntimeConfig,
    InferenceStageConfig,
    RenderConfig,
    RoutingConfig,
    ValidationConfig,
)


def inference_stage_recipe(stage: str) -> InferenceStageConfig:
    render = RenderConfig()
    routing = RoutingConfig()
    validation = ValidationConfig()
    assembly = AssemblyConfig()
    runtime = InferenceRuntimeConfig()
    shared_modes = (
        DecodeModeConfig(
            name="fast",
            visual_token_budget=96,
            max_output_tokens=3072,
            crop_budget=0,
            repair_budget=0,
            grammar_mode="light",
            structured_output_backend="auto",
            allow_block_local_repair=False,
        ),
        DecodeModeConfig(
            name="standard",
            visual_token_budget=160,
            max_output_tokens=5120,
            crop_budget=2,
            repair_budget=1,
            grammar_mode="light",
            structured_output_backend="xgrammar",
            allow_block_local_repair=True,
            specialist_hints=("table",),
        ),
        DecodeModeConfig(
            name="hard",
            visual_token_budget=256,
            max_output_tokens=8192,
            crop_budget=4,
            repair_budget=2,
            grammar_mode="strict",
            structured_output_backend="xgrammar",
            allow_block_local_repair=True,
            specialist_hints=("formula", "handwriting", "table"),
        ),
    )

    recipes = {
        "i1_vllm_adaptive": InferenceStageConfig(
            stage="i1_vllm_adaptive",
            artifact_variant_id="llmcompressor_fp8_dynamic",
            artifact_uri="hf://leopardi-ocr-checkpoints/leo-s0-o2/llmcompressor_fp8_dynamic",
            runtime_family="vllm",
            fallback_runtime_family="sglang",
            structured_backend_default="xgrammar",
            render=render,
            routing=routing,
            validation=validation,
            assembly=assembly,
            runtime=runtime,
            modes=shared_modes,
        ),
        "i2_sglang_structured": InferenceStageConfig(
            stage="i2_sglang_structured",
            artifact_variant_id="llmcompressor_fp8_dynamic",
            artifact_uri="hf://leopardi-ocr-checkpoints/leo-s0-o2/llmcompressor_fp8_dynamic",
            runtime_family="sglang",
            fallback_runtime_family="vllm",
            structured_backend_default="xgrammar",
            render=render,
            routing=routing,
            validation=validation,
            assembly=AssemblyConfig(emit_page_break_markers=True),
            runtime=runtime,
            modes=shared_modes,
        ),
    }
    if stage not in recipes:
        raise KeyError(f"Unknown inference stage recipe: {stage}")
    return recipes[stage]


def inference_stage_recipe_dict(stage: str) -> dict[str, object]:
    return asdict(inference_stage_recipe(stage))
