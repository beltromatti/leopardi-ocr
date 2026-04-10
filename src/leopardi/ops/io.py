from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping

from leopardi.ops.layout import RunLayout, build_run_layout
from leopardi.ops.schemas import RunHeartbeat, RunManifest, RunSummary


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def ensure_run_layout(
    experiment_id: str | None = None,
    *,
    root: str | Path = "runs",
    layout: RunLayout | None = None,
) -> RunLayout:
    if layout is None:
        if experiment_id is None:
            raise ValueError("experiment_id is required when layout is not provided")
        layout = build_run_layout(experiment_id, root=root)
    for directory in (
        layout.experiment_root,
        layout.control_dir,
        layout.artifacts_dir,
        layout.scratch_dir,
    ):
        directory.mkdir(parents=True, exist_ok=True)
    return layout


def write_manifest(manifest: RunManifest, *, layout: RunLayout | None = None) -> Path:
    run_layout = ensure_run_layout(
        experiment_id=manifest.experiment_id,
        root=manifest.local_run_root.rsplit(f"/{manifest.experiment_id}", 1)[0]
        if manifest.local_run_root.endswith(f"/{manifest.experiment_id}")
        else "runs",
        layout=layout,
    )
    _write_json(run_layout.manifest_path, manifest.model_dump())
    return run_layout.manifest_path


def write_heartbeat(
    heartbeat: RunHeartbeat,
    *,
    layout: RunLayout,
) -> Path:
    ensure_run_layout(layout=layout)
    payload = heartbeat.model_dump()
    payload["updated_at"] = _utc_now()
    _write_json(layout.heartbeat_path, payload)
    return layout.heartbeat_path


def write_summary(summary: RunSummary, *, layout: RunLayout) -> Path:
    ensure_run_layout(layout=layout)
    payload = summary.model_dump()
    payload["written_at"] = _utc_now()
    _write_json(layout.summary_path, payload)
    return layout.summary_path


def append_event(
    *,
    layout: RunLayout,
    event_type: str,
    phase: str,
    stage: str,
    payload: Mapping[str, Any] | None = None,
) -> Path:
    ensure_run_layout(layout=layout)
    event = {
        "ts": _utc_now(),
        "event_type": event_type,
        "phase": phase,
        "stage": stage,
        "payload": dict(payload or {}),
    }
    with layout.events_path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(event, sort_keys=True) + "\n")
    return layout.events_path


def append_console_log(
    *,
    layout: RunLayout,
    message: str,
) -> Path:
    ensure_run_layout(layout=layout)
    with layout.console_log_path.open("a", encoding="utf-8") as handle:
        handle.write(f"{_utc_now()} {message.rstrip()}\n")
    return layout.console_log_path


def stop_requested(layout: RunLayout) -> bool:
    return layout.stop_flag_path.exists()


def reload_requested(layout: RunLayout) -> bool:
    return layout.reload_flag_path.exists()


def read_note(layout: RunLayout) -> str | None:
    if not layout.note_path.exists():
        return None
    note = layout.note_path.read_text(encoding="utf-8").strip()
    return note or None
