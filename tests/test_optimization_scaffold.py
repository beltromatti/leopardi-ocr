from __future__ import annotations

from leopardi.optimization import (
    OptimizationGoalConfig,
    OptimizationStageConfig,
    VariantMeasurement,
    build_variant_plan,
    build_variant_runtime_plan,
    materialize_optimization_stage,
    pareto_frontier,
    rank_candidates,
)


def test_optimization_config_and_plan() -> None:
    stage = OptimizationStageConfig(
        stage="o2_vllm_compressed",
        variants=(),
    )
    assert stage.stage == "o2_vllm_compressed"

    populated = OptimizationStageConfig.from_dict(
        {
            "stage": "o1_torchao_portable",
            "variants": [
                {
                    "variant_id": "torchao_int4_weight_only",
                    "method": "torchao_ptq",
                    "runtime_targets": ["hf", "vllm"],
                    "requires_calibration": False,
                }
            ],
        }
    )
    plan = build_variant_plan(populated)

    assert len(plan) == 1
    assert plan[0]["variant_id"] == "torchao_int4_weight_only"
    assert plan[0]["runtime_targets"] == ["hf", "vllm"]


def test_optimization_ranking_and_frontier() -> None:
    goal = OptimizationGoalConfig()
    reference = VariantMeasurement(
        variant_id="bf16_reference",
        overall_score=0.945,
        markdown_validity=0.998,
        latex_validity=0.994,
        table_score=0.928,
        formula_score=0.941,
        latency_ms=1200.0,
        peak_memory_gb=22.0,
        throughput_pages_per_second=0.83,
    )
    candidates = [
        VariantMeasurement(
            variant_id="llmcompressor_fp8_dynamic",
            overall_score=0.941,
            markdown_validity=0.997,
            latex_validity=0.992,
            table_score=0.925,
            formula_score=0.938,
            latency_ms=930.0,
            peak_memory_gb=16.5,
            throughput_pages_per_second=1.08,
        ),
        VariantMeasurement(
            variant_id="torchao_int4_weight_only",
            overall_score=0.934,
            markdown_validity=0.996,
            latex_validity=0.99,
            table_score=0.919,
            formula_score=0.931,
            latency_ms=760.0,
            peak_memory_gb=12.0,
            throughput_pages_per_second=1.31,
        ),
    ]

    ranked = rank_candidates(reference, candidates, goal)
    frontier = pareto_frontier([reference, *candidates])

    assert ranked[0].passes_floors is True
    assert ranked[0].score >= ranked[1].score
    assert all(item.passes_floors for item in ranked)
    assert any(item.variant_id == "llmcompressor_fp8_dynamic" for item in frontier)
    assert any(item.variant_id == "torchao_int4_weight_only" for item in frontier)


def test_optimization_runtime_plan_and_materialization(tmp_path) -> None:
    stage = OptimizationStageConfig.from_dict(
        {
            "stage": "o2_vllm_compressed",
            "base_stage": "f3_rlvr",
            "variants": [
                {
                    "variant_id": "llmcompressor_fp8_dynamic",
                    "method": "llmcompressor_fp8",
                    "export_format": "compressed_tensors",
                    "runtime_targets": ["vllm"],
                }
            ],
        },
        runtime_payload={"runtime": {"hardware_tag": "rtx5090"}},
    )
    plan, card = build_variant_runtime_plan(
        experiment_id="leo-s0-o2-test",
        stage=stage,
        variant=stage.variants[0],
        artifacts_root=tmp_path / "artifacts",
        persistent_root="hf://leopardi-ocr-checkpoints",
        base_checkpoint_uri="hf://leopardi/base",
    )

    assert plan.backend_family == "llmcompressor"
    assert "LEOPARDI_BASE_CHECKPOINT_DIR" in plan.commands[0]
    assert "llmcompressor" in plan.commands[1]
    assert card.persistent_artifact_uri.endswith("/leo-s0-o2-test/o2_vllm_compressed/llmcompressor_fp8_dynamic")

    payload = materialize_optimization_stage(
        experiment_id="leo-s0-o2-test",
        stage=stage,
        base_checkpoint_uri="hf://leopardi/base",
        root=tmp_path / "runs",
    )
    assert payload["plans"][0]["variant_id"] == "llmcompressor_fp8_dynamic"
    artifact_card_path = tmp_path / "runs" / "leo-s0-o2-test" / "artifacts" / "optimization" / "llmcompressor_fp8_dynamic" / "artifact-card.json"
    assert artifact_card_path.exists()
