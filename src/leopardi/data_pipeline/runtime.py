from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path

from leopardi.data_pipeline.config import DataBuildStageConfig
from leopardi.data_pipeline.planner import build_data_build_execution_plan, plan_dict
from leopardi.data_pipeline.registry import registry_summary
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


def _write_json(path: Path, payload: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def materialize_data_build_stage(
    *,
    experiment_id: str,
    stage: DataBuildStageConfig,
    stage_config_path: str | None = None,
    runtime_config_path: str | None = None,
    root: str | Path = "runs",
) -> dict[str, object]:
    layout = ensure_run_layout(experiment_id, root=root)
    plan = build_data_build_execution_plan(
        experiment_id=experiment_id,
        stage=stage,
        stage_config_path=stage_config_path or f"generated::data::{stage.stage}",
        runtime_config_path=runtime_config_path or "generated::runtime::data_build",
        root=root,
    )

    write_manifest(
        RunManifest(
            experiment_id=experiment_id,
            phase="data_pipeline",
            stage=stage.stage,
            track=stage.profile_id,
            hardware_tag=stage.runtime.hardware_tag,
            config_paths=[
                stage_config_path or f"generated::data::{stage.stage}",
                runtime_config_path or "generated::runtime::data_build",
            ],
            data_bundle_ids=list(plan.bundle_ids),
            protocol_version="data_pipeline_v1",
            local_run_root=str(layout.experiment_root),
            persistent_targets={
                "bundles": stage.runtime.persistence.bundle_target,
                "metadata": stage.runtime.persistence.metadata_target,
            },
        ),
        layout=layout,
    )
    write_heartbeat(
        RunHeartbeat(
            experiment_id=experiment_id,
            phase="data_pipeline",
            stage=stage.stage,
            state="draft",
            current_step=0,
        ),
        layout=layout,
    )

    _write_json(Path(plan.plan_path), plan_dict(plan))
    _write_json(
        Path(plan.report_stub_path),
        {
            "experiment_id": experiment_id,
            "stage": stage.stage,
            "status": "pending_build",
            "profile_id": stage.profile_id,
            "focus": [
                "bounded_disk_usage",
                "publish_then_purge",
                "remote_reuse_on_ephemeral_machines",
            ],
        },
    )
    _write_json(
        Path(plan.local_paths.publish_ledger_path),
        {
            "experiment_id": experiment_id,
            "stage": stage.stage,
            "profile_id": stage.profile_id,
            "upload_mode": stage.runtime.persistence.upload_mode,
            "verification_required": stage.publish_verify_required,
            "bundles": [
                {
                    "bundle_id": spec.bundle_id,
                    "sample_uri": spec.sample_uri,
                    "bundle_uri": spec.bundle_uri,
                    "manifest_uri": spec.manifest_uri,
                    "status": "queued",
                }
                for spec in plan.bundle_specs
            ],
        },
    )

    for bundle in plan.bundle_specs:
        _write_json(
            Path(bundle.local_manifest_dir) / "bundle-card.stub.json",
            {
                "bundle_id": bundle.bundle_id,
                "stage": bundle.stage,
                "bundle_class": bundle.bundle_class,
                "source_ids": list(bundle.source_ids),
                "sample_artifact_group": bundle.sample_artifact_group,
                "sample_uri": bundle.sample_uri,
                "bundle_uri": bundle.bundle_uri,
                "manifest_uri": bundle.manifest_uri,
                "retention_mode": bundle.retention_mode,
            },
        )

    append_event(
        layout=layout,
        event_type="data_pipeline_plan_materialized",
        phase="data_pipeline",
        stage=stage.stage,
        payload={
            "profile_id": stage.profile_id,
            "bundle_count": len(plan.bundle_specs),
            "source_count": len(plan.source_ids),
        },
    )
    write_summary(
        RunSummary(
            experiment_id=experiment_id,
            phase="data_pipeline",
            stage=stage.stage,
            outcome="completed",
            key_metrics={
                "bundle_count": float(len(plan.bundle_specs)),
                "source_count": float(len(plan.source_ids)),
                "source_wave_count": float(len(plan.source_waves)),
            },
            artifacts=[
                ArtifactPointer(
                    artifact_kind="runtime_plan",
                    uri=f"local://{plan.plan_path}",
                    local_path=plan.plan_path,
                    persistence_status="local_only",
                ),
                ArtifactPointer(
                    artifact_kind="summary_table",
                    uri=f"local://{plan.local_paths.publish_ledger_path}",
                    local_path=plan.local_paths.publish_ledger_path,
                    persistence_status="local_only",
                ),
                ArtifactPointer(
                    artifact_kind="bundle",
                    uri=stage.runtime.persistence.bundle_target,
                    local_path=plan.local_paths.upload_staging_dir,
                    persistence_status="queued",
                ),
            ],
            notes=[
                "Data-pipeline control-plane artifacts materialized successfully.",
                "Use source waves and publish ledger as the single source of truth on rented RTX 5090 builders.",
            ],
        ),
        layout=layout,
    )
    return {
        "layout": layout.as_dict(),
        "plan": asdict(plan),
        "registry": registry_summary(),
    }
