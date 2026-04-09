from __future__ import annotations

from leopardi.evaluation import (
    EvaluationSample,
    EvaluationStageConfig,
    compile_evaluation_result,
    load_baseline_registry,
    load_dataset_registry,
    materialize_evaluation_stage,
    normalize_markdown,
    registry_summary,
    write_evaluation_report,
)


def test_evaluation_config_and_registry() -> None:
    stage = EvaluationStageConfig.from_dict(
        {
            "eval": {
                "protocol": "public_frontier",
                "public_benchmarks": ["omnidocbench_v15", "olmocr_bench"],
                "required_slices": ["formulas", "tables"],
            }
        },
        runtime_payload={"runtime": {"hardware_tag": "rtx5090", "primary_runtime": "vllm"}},
    )
    assert stage.bundle_id == "public_frontier_v1"
    assert stage.protocol == "public_frontier_v1"
    assert stage.decode_modes == ("fast", "standard", "hard")

    summary = registry_summary()
    assert summary["dataset_count"] > 0
    assert summary["baseline_count"] > 0
    assert "OmniDocBench_v15" in summary["public_dataset_families"]


def test_evaluation_result_compilation() -> None:
    stage = EvaluationStageConfig.from_dict(
        {"eval": {"protocol": "public_frontier"}},
        runtime_payload={"runtime": {"hardware_tag": "rtx5090", "structured_output_backend": "auto"}},
    )
    samples = [
        EvaluationSample(
            sample_id="sample-1",
            dataset_family="OmniDocBench_v15",
            decode_mode="standard",
            prediction_markdown="# A\n\n| x | y |\n| --- | --- |\n| 1 | 2 |",
            reference_markdown="# A\n\n| x | y |\n| --- | --- |\n| 1 | 2 |",
            latency_ms=1200.0,
            output_tokens=320,
            formula_prediction="x+y",
            formula_reference="x+y",
            native_metrics={
                "page_overall": 0.95,
                "table_teds": 0.93,
                "rotation_score": 0.98,
                "wild_page_score": 0.9,
                "vram_peak_gib": 18.0,
            },
            protocol_version=stage.protocol,
        ),
        EvaluationSample(
            sample_id="sample-2",
            dataset_family="olmOCR_Bench",
            decode_mode="standard",
            prediction_markdown=normalize_markdown("## B\n\nText"),
            reference_markdown="## B\n\nText",
            latency_ms=1260.0,
            output_tokens=290,
            native_metrics={
                "page_overall": 0.94,
                "rotation_score": 0.97,
                "wild_page_score": 0.89,
                "vram_peak_gib": 18.2,
            },
            protocol_version=stage.protocol,
        ),
    ]
    result = compile_evaluation_result(
        experiment_id="leo-s0-eval-test",
        stage=stage,
        runtime_family="vllm",
        decode_mode="standard",
        model_name="Leopardi-S0",
        size_band="~150M",
        evidence_grade="local_reproduction",
        samples=samples,
        datasets=load_dataset_registry(),
        baselines=load_baseline_registry(),
        params_total_b=0.093,
        lus=1.41,
    )

    assert result.aggregate_metrics["pages_per_second"] > 0.0
    assert any(row.key == "Leopardi-S0" for row in result.scorecards)
    assert len(result.failure_slices) == 6
    assert result.report_package.protocol_version == "public_frontier_v1"
    assert result.report_package.dataset_bundle_summary["bundle_id"] == "public_frontier_v1"


def test_evaluation_materialization_and_report(tmp_path) -> None:
    stage = EvaluationStageConfig.from_dict(
        {"eval": {"protocol": "public_frontier"}},
        runtime_payload={"runtime": {"hardware_tag": "rtx5090", "primary_runtime": "vllm", "alternate_runtime": "sglang"}},
    )
    payload = materialize_evaluation_stage(
        experiment_id="leo-s0-eval-test",
        stage=stage,
        root=tmp_path / "runs",
    )
    assert payload["plans"][0]["runtime_family"] == "vllm"
    assert (
        tmp_path
        / "runs"
        / "leo-s0-eval-test"
        / "artifacts"
        / "evaluation"
        / "public_frontier_v1"
        / "vllm"
        / "report.json"
    ).exists()

    result = write_evaluation_report(
        experiment_id="leo-s0-eval-test",
        stage=stage,
        runtime_family="vllm",
        decode_mode="hard",
        samples=[
            EvaluationSample(
                sample_id="sample-1",
                dataset_family="Real5_OmniDocBench",
                decode_mode="hard",
                prediction_markdown="# Scan\n\nText",
                reference_markdown="# Scan\n\nText",
                latency_ms=1500.0,
                output_tokens=360,
                native_metrics={"page_overall": 0.94, "rotation_score": 0.98, "wild_page_score": 0.92},
                protocol_version=stage.protocol,
            )
        ],
        model_name="Leopardi-S0",
        size_band="~150M",
        params_total_b=0.093,
        root=tmp_path / "runs",
    )
    assert result["result"]["report_package"]["runtime_family"] == "vllm"
    assert (
        tmp_path
        / "runs"
        / "leo-s0-eval-test"
        / "artifacts"
        / "evaluation"
        / "public_frontier_v1"
        / "vllm"
        / "normalized-predictions.jsonl"
    ).exists()
