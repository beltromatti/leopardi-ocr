from __future__ import annotations

import json

from leopardi.ops import (
    RunHeartbeat,
    RunManifest,
    RunSummary,
    append_event,
    ensure_run_layout,
    read_note,
    reload_requested,
    stop_requested,
    write_heartbeat,
    write_manifest,
    write_summary,
)


def test_ops_io_materializes_expected_files(tmp_path) -> None:
    layout = ensure_run_layout("leo-s0-p2-dense-20260408-001", root=tmp_path)
    manifest = RunManifest(
        experiment_id="leo-s0-p2-dense-20260408-001",
        phase="pretraining",
        stage="p2_multimodal_core",
        track="s0-core",
        hardware_tag="rtx5090",
        config_paths=["configs/model/leopardi_s0.yaml"],
        local_run_root=str(layout.experiment_root),
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
    )

    write_manifest(manifest, layout=layout)
    write_heartbeat(heartbeat, layout=layout)
    write_summary(summary, layout=layout)
    append_event(
        layout=layout,
        event_type="checkpoint_saved",
        phase="pretraining",
        stage="p2_multimodal_core",
        payload={"step": 1000},
    )

    assert layout.manifest_path.exists()
    assert layout.heartbeat_path.exists()
    assert layout.summary_path.exists()
    event = json.loads(layout.events_path.read_text(encoding="utf-8").strip())
    assert event["event_type"] == "checkpoint_saved"
    assert event["payload"]["step"] == 1000


def test_ops_control_helpers_reflect_files(tmp_path) -> None:
    layout = ensure_run_layout("leo-s0-p2-dense-20260408-001", root=tmp_path)
    layout.stop_flag_path.write_text("", encoding="utf-8")
    layout.reload_flag_path.write_text("", encoding="utf-8")
    layout.note_path.write_text("reduce table curriculum", encoding="utf-8")

    assert stop_requested(layout) is True
    assert reload_requested(layout) is True
    assert read_note(layout) == "reduce table curriculum"
