from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path

from leopardi.evaluation.config import EvaluationStageConfig
from leopardi.evaluation.metrics import EvaluationSample
from leopardi.evaluation.pipeline import compile_evaluation_result, evaluation_result_bundle_dict
from leopardi.evaluation.registry import load_baseline_registry, load_dataset_registry
from leopardi.evaluation.reports import report_package_dict
from leopardi.ops import (
    ArtifactPointer,
    RunHeartbeat,
    RunManifest,
    RunSummary,
    append_event,
    ensure_run_layout,
    write_heartbeat,
    write_manifest,
    write_summary,
)


@dataclass(slots=True)
class EvaluationExecutionPlan:
    protocol: str
    bundle_id: str
    decode_modes: tuple[str, ...]
    runtime_family: str
    sample_manifest_path: str
    normalized_predictions_path: str
    scorecard_path: str
    report_stub_path: str
    report_path: str
    competitor_table_path: str
    report_uri: str


def _write_json(path: Path, payload: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def build_execution_plan(
    *,
    experiment_id: str,
    stage: EvaluationStageConfig,
    runtime_family: str,
    artifacts_root: str | Path,
    persistent_report_root: str,
) -> EvaluationExecutionPlan:
    artifacts_root = Path(artifacts_root)
    root = artifacts_root / "evaluation" / stage.protocol / runtime_family
    return EvaluationExecutionPlan(
        protocol=stage.protocol,
        bundle_id=stage.bundle_id,
        decode_modes=stage.decode_modes,
        runtime_family=runtime_family,
        sample_manifest_path=str(root / "samples.json"),
        normalized_predictions_path=str(root / "normalized-predictions.jsonl"),
        scorecard_path=str(root / "scorecards.json"),
        report_stub_path=str(root / "report.stub.json"),
        report_path=str(root / "report.json"),
        competitor_table_path=str(root / "competitors.json"),
        report_uri=f"{persistent_report_root.rstrip('/')}/{experiment_id}/{stage.protocol}/{runtime_family}",
    )


def write_evaluation_report(
    *,
    experiment_id: str,
    stage: EvaluationStageConfig,
    runtime_family: str,
    decode_mode: str,
    samples: list[EvaluationSample],
    model_name: str,
    size_band: str,
    params_total_b: float,
    lus: float = 0.0,
    evidence_grade: str = "local_reproduction",
    root: str | Path = "runs",
) -> dict[str, object]:
    layout = ensure_run_layout(experiment_id, root=root)
    plan = build_execution_plan(
        experiment_id=experiment_id,
        stage=stage,
        runtime_family=runtime_family,
        artifacts_root=layout.artifacts_dir,
        persistent_report_root="hf://leopardi-ocr-reports",
    )
    datasets = load_dataset_registry()
    baselines = load_baseline_registry()
    result = compile_evaluation_result(
        experiment_id=experiment_id,
        stage=stage,
        runtime_family=runtime_family,
        decode_mode=decode_mode,
        model_name=model_name,
        size_band=size_band,
        evidence_grade=evidence_grade,
        samples=samples,
        datasets=datasets,
        baselines=baselines,
        params_total_b=params_total_b,
        lus=lus,
        artifact_pointers=[
            {
                "artifact_kind": "report",
                "uri": plan.report_uri,
                "local_path": plan.report_path,
                "persistence_status": "queued",
            }
        ],
        evidence_notes=[
            "Generated from local synthetic scoring inputs.",
            "Replace synthetic inputs with benchmark adapters before using this report for claims.",
        ],
    )
    report_payload = report_package_dict(result.report_package)
    _write_json(Path(plan.scorecard_path), {"scorecards": [row.values for row in result.scorecards]})
    _write_json(Path(plan.report_path), report_payload)
    normalized_path = Path(plan.normalized_predictions_path)
    normalized_path.parent.mkdir(parents=True, exist_ok=True)
    normalized_path.write_text(
        "".join(
            json.dumps(
                {
                    "sample_id": sample.sample_id,
                    "dataset_family": sample.dataset_family,
                    "decode_mode": sample.decode_mode,
                    "prediction_markdown": sample.prediction_markdown,
                    "reference_markdown": sample.reference_markdown,
                },
                sort_keys=True,
            )
            + "\n"
            for sample in samples
        ),
        encoding="utf-8",
    )
    append_event(
        layout=layout,
        event_type="evaluation_report_written",
        phase="evaluation",
        stage=stage.protocol,
        payload={
            "runtime_family": runtime_family,
            "decode_mode": decode_mode,
            "report_path": plan.report_path,
        },
    )
    return {
        "plan": asdict(plan),
        "result": evaluation_result_bundle_dict(result),
    }


def materialize_evaluation_stage(
    *,
    experiment_id: str,
    stage: EvaluationStageConfig,
    stage_config_path: str | None = None,
    runtime_config_path: str | None = None,
    root: str | Path = "runs",
) -> dict[str, object]:
    layout = ensure_run_layout(experiment_id, root=root)
    write_manifest(
        RunManifest(
            experiment_id=experiment_id,
            phase="evaluation",
            stage=stage.protocol,
            track="evaluation",
            hardware_tag=stage.runtime.hardware_tag,
            config_paths=[
                stage_config_path or f"generated::evaluation::{stage.protocol}",
                runtime_config_path or "generated::runtime::evaluation",
            ],
            data_bundle_ids=[stage.bundle_id],
            protocol_version=stage.protocol,
            local_run_root=str(layout.experiment_root),
            persistent_targets={
                "reports": "hf://leopardi-ocr-reports",
                "metadata": "hf://leopardi-ocr-metadata",
            },
        ),
        layout=layout,
    )
    write_heartbeat(
        RunHeartbeat(
            experiment_id=experiment_id,
            phase="evaluation",
            stage=stage.protocol,
            state="draft",
            current_step=0,
        ),
        layout=layout,
    )

    runtime_families = dict.fromkeys((stage.runtime.primary_runtime, stage.runtime.alternate_runtime))
    plans = [
        build_execution_plan(
            experiment_id=experiment_id,
            stage=stage,
            runtime_family=runtime_family,
            artifacts_root=layout.artifacts_dir,
            persistent_report_root="hf://leopardi-ocr-reports",
        )
        for runtime_family in runtime_families
    ]
    datasets = load_dataset_registry()
    baselines = load_baseline_registry()

    for plan in plans:
        _write_json(
            Path(plan.sample_manifest_path),
            {
                "protocol": stage.protocol,
                "bundle_id": stage.bundle_id,
                "decode_modes": list(stage.decode_modes),
                "runtime_family": plan.runtime_family,
                "datasets": [
                    entry.dataset_family for entry in datasets if stage.protocol in entry.covered_protocols
                ],
            },
        )
        _write_json(
            Path(plan.scorecard_path),
            {
                "protocol": stage.protocol,
                "required_scorecards": [
                    "public_frontier" if stage.protocol.startswith("public_frontier") else "internal_promotion",
                    "failure_slice",
                    "size_normalized",
                ],
            },
        )
        _write_json(
            Path(plan.competitor_table_path),
            {
                "baselines": [
                    {
                        "baseline_id": item.baseline_id,
                        "model_name": item.model_name,
                        "size_band": item.size_band,
                        "evidence_policy": item.evidence_policy,
                    }
                    for item in baselines
                    if stage.protocol in item.main_protocols
                ]
            },
        )
        _write_json(
            Path(plan.report_stub_path),
            {
                "experiment_id": experiment_id,
                "protocol": stage.protocol,
                "bundle_id": stage.bundle_id,
                "runtime_family": plan.runtime_family,
                "status": "pending_execution",
                "decode_modes": list(stage.decode_modes),
            },
        )
        _write_json(
            Path(plan.report_path),
            {
                "experiment_id": experiment_id,
                "protocol": stage.protocol,
                "bundle_id": stage.bundle_id,
                "runtime_family": plan.runtime_family,
                "status": "draft",
                "note": "Populate this file with write_evaluation_report after benchmark execution.",
            },
        )
        append_event(
            layout=layout,
            event_type="evaluation_plan_materialized",
            phase="evaluation",
            stage=stage.protocol,
            payload={
                "runtime_family": plan.runtime_family,
                "bundle_id": stage.bundle_id,
            },
        )

    write_summary(
        RunSummary(
            experiment_id=experiment_id,
            phase="evaluation",
            stage=stage.protocol,
            outcome="completed",
            key_metrics={},
            artifacts=[
                ArtifactPointer(
                    artifact_kind="report",
                    uri=plan.report_uri,
                    local_path=plan.report_path,
                    persistence_status="queued",
                )
                for plan in plans
            ]
            + [
                ArtifactPointer(
                    artifact_kind="runtime_plan",
                    uri=f"local://{plan.sample_manifest_path}",
                    local_path=plan.sample_manifest_path,
                    persistence_status="local_only",
                )
                for plan in plans
            ],
            notes=[
                "Evaluation execution plans materialized successfully.",
                "Use inference launch plans plus benchmark adapters and write_evaluation_report to fill the generated report artifacts.",
            ],
        ),
        layout=layout,
    )
    return {
        "layout": layout.as_dict(),
        "plans": [asdict(plan) for plan in plans],
        "dataset_count": len(datasets),
        "baseline_count": len(baselines),
    }
