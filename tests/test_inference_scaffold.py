from __future__ import annotations

from leopardi.inference import (
    DocumentPage,
    InferenceStageConfig,
    PageSignals,
    assemble_document,
    build_launch_plan,
    materialize_inference_stage,
    repair_required,
    route_page,
    validate_markdown,
)


def test_inference_routing_and_validation() -> None:
    stage = InferenceStageConfig.from_dict(
        {
            "stage": "i1_vllm_adaptive",
            "modes": [
                {
                    "name": "fast",
                    "visual_token_budget": 96,
                    "max_output_tokens": 3072,
                    "crop_budget": 0,
                    "repair_budget": 0,
                },
                {
                    "name": "standard",
                    "visual_token_budget": 160,
                    "max_output_tokens": 5120,
                    "crop_budget": 2,
                    "repair_budget": 1,
                },
                {
                    "name": "hard",
                    "visual_token_budget": 256,
                    "max_output_tokens": 8192,
                    "crop_budget": 4,
                    "repair_budget": 2,
                    "specialist_hints": ["formula"],
                },
            ],
        },
        runtime_payload={"runtime": {"hardware_tag": "rtx5090"}},
    )
    decision = route_page(
        stage,
        PageSignals(
            visual_density=0.65,
            block_count_estimate=20,
            formula_density=0.14,
            table_density=0.05,
            handwriting_likelihood=0.05,
            chart_likelihood=0.01,
            long_tiny_text_likelihood=0.12,
            photo_distortion_likelihood=0.02,
            orientation_uncertainty=0.05,
        ),
    )
    assert decision.mode == "hard"
    assert "formula" in decision.specialist_hints

    report = validate_markdown("```python\nprint('x')\n", stage.validation)
    assert report.valid is False
    assert repair_required(report, stage) is True


def test_inference_materialization_and_assembly(tmp_path) -> None:
    stage = InferenceStageConfig.from_dict(
        {
            "stage": "i1_vllm_adaptive",
            "artifact_variant_id": "llmcompressor_fp8_dynamic",
            "artifact_uri": "hf://leopardi-ocr-checkpoints/leo-s0-o2/llmcompressor_fp8_dynamic",
            "runtime_family": "vllm",
            "fallback_runtime_family": "sglang",
            "modes": [
                {
                    "name": "fast",
                    "visual_token_budget": 96,
                    "max_output_tokens": 3072,
                    "crop_budget": 0,
                    "repair_budget": 0,
                },
                {
                    "name": "standard",
                    "visual_token_budget": 160,
                    "max_output_tokens": 5120,
                    "crop_budget": 2,
                    "repair_budget": 1,
                },
                {
                    "name": "hard",
                    "visual_token_budget": 256,
                    "max_output_tokens": 8192,
                    "crop_budget": 4,
                    "repair_budget": 2,
                },
            ],
        },
        runtime_payload={"runtime": {"hardware_tag": "rtx5090"}},
    )
    plan = build_launch_plan(
        experiment_id="leo-s0-i1-test",
        stage=stage,
        runtime_family="vllm",
        artifacts_root=tmp_path / "artifacts",
        persistent_report_root="hf://leopardi-ocr-reports",
    )
    assert "vllm serve" in plan.launch_command
    assert plan.runtime_report_uri.endswith("/leo-s0-i1-test/i1_vllm_adaptive/vllm")

    payload = materialize_inference_stage(
        experiment_id="leo-s0-i1-test",
        stage=stage,
        root=tmp_path / "runs",
    )
    assert payload["plans"][0]["runtime_family"] == "vllm"
    assert (
        tmp_path
        / "runs"
        / "leo-s0-i1-test"
        / "artifacts"
        / "inference"
        / "vllm"
        / "runtime-plan.json"
    ).exists()

    document = assemble_document(
        [
            DocumentPage(page_number=1, markdown="Shared Header\n\n# A\n\nAlpha\n\nShared Footer"),
            DocumentPage(page_number=2, markdown="Shared Header\n\nBeta\n\nShared Footer"),
        ],
        stage.assembly,
    )
    assert "Shared Header" not in document
    assert "Shared Footer" not in document
