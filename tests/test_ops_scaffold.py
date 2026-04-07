from __future__ import annotations

from leopardi.ops import ArtifactPointer, RunHeartbeat, RunManifest, RunSummary, build_run_layout


def test_run_layout_contains_expected_paths() -> None:
    layout = build_run_layout("leo-s0-p2-dense-20260408-001")
    payload = layout.as_dict()

    assert payload["experiment_root"].endswith("leo-s0-p2-dense-20260408-001")
    assert payload["heartbeat_path"].endswith("heartbeat.json")
    assert payload["stop_flag_path"].endswith("control/STOP")


def test_ops_schema_roundtrip() -> None:
    manifest = RunManifest(
        experiment_id="leo-s0-p2-dense-20260408-001",
        phase="pretraining",
        stage="p2_multimodal_core",
        track="s0-core",
        hardware_tag="rtx5090",
        config_paths=["configs/model/leopardi_s0.yaml"],
        local_run_root="runs/leo-s0-p2-dense-20260408-001",
    )
    heartbeat = RunHeartbeat(
        experiment_id=manifest.experiment_id,
        phase=manifest.phase,
        stage=manifest.stage,
        state="running",
    )
    summary = RunSummary(
        experiment_id=manifest.experiment_id,
        phase=manifest.phase,
        stage=manifest.stage,
        outcome="completed",
        artifacts=[
            ArtifactPointer(
                artifact_kind="checkpoint",
                uri="hf://leopardi-ocr-checkpoints/leo-s0-p2-dense-20260408-001",
                persistence_status="published",
            )
        ],
    )

    assert manifest.model_dump()["phase"] == "pretraining"
    assert heartbeat.model_dump()["state"] == "running"
    assert summary.artifacts[0].artifact_kind == "checkpoint"
