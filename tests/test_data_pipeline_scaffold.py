from __future__ import annotations

from leopardi.data_pipeline import (
    DataBuildStageConfig,
    build_data_build_execution_plan,
    materialize_data_build_stage,
    registry_summary,
)


def test_data_pipeline_registry_summary_smoke() -> None:
    summary = registry_summary()
    assert summary["source_count"] >= 10
    assert "arxiv_source_pdf" in summary["exact_foundations"]


def test_data_pipeline_plan_exact_core(tmp_path) -> None:
    stage = DataBuildStageConfig.from_yaml(
        "configs/data/s0_exact_core_build.yaml",
        "configs/runtime/data_build_rtx5090.yaml",
    )
    plan = build_data_build_execution_plan(
        experiment_id="leo-s0-data-exact-core",
        stage=stage,
        stage_config_path="configs/data/s0_exact_core_build.yaml",
        runtime_config_path="configs/runtime/data_build_rtx5090.yaml",
        root=tmp_path / "runs",
    )

    assert plan.profile_id == "exact_core_only"
    assert plan.bundle_ids == ("tokenizer_v1", "p1_text_warmup_v1", "p2_exact_core_v1")
    assert plan.source_ids == ("arxiv_source_pdf", "pmc_oa_pdf_xml")
    assert len(plan.source_waves) == 1
    assert len(plan.bundle_specs) == 3


def test_data_pipeline_materialization(tmp_path) -> None:
    stage = DataBuildStageConfig.from_yaml(
        "configs/data/s0_exact_core_build.yaml",
        "configs/runtime/data_build_rtx5090.yaml",
    )
    payload = materialize_data_build_stage(
        experiment_id="leo-s0-data-smoke-20260408-001",
        stage=stage,
        stage_config_path="configs/data/s0_exact_core_build.yaml",
        runtime_config_path="configs/runtime/data_build_rtx5090.yaml",
        root=tmp_path / "runs",
    )

    assert payload["plan"]["stage"] == "s0_exact_core_build"
    assert (
        tmp_path
        / "runs"
        / "leo-s0-data-smoke-20260408-001"
        / "artifacts"
        / "data_pipeline"
        / "s0_exact_core_build"
        / "data-build-plan.json"
    ).exists()
