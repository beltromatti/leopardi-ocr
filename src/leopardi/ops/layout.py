from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


@dataclass(slots=True)
class RunLayout:
    root: Path
    experiment_root: Path
    manifest_path: Path
    heartbeat_path: Path
    summary_path: Path
    console_log_path: Path
    events_path: Path
    control_dir: Path
    stop_flag_path: Path
    reload_flag_path: Path
    note_path: Path
    artifacts_dir: Path
    scratch_dir: Path

    def as_dict(self) -> dict[str, str]:
        return {
            "root": str(self.root),
            "experiment_root": str(self.experiment_root),
            "manifest_path": str(self.manifest_path),
            "heartbeat_path": str(self.heartbeat_path),
            "summary_path": str(self.summary_path),
            "console_log_path": str(self.console_log_path),
            "events_path": str(self.events_path),
            "control_dir": str(self.control_dir),
            "stop_flag_path": str(self.stop_flag_path),
            "reload_flag_path": str(self.reload_flag_path),
            "note_path": str(self.note_path),
            "artifacts_dir": str(self.artifacts_dir),
            "scratch_dir": str(self.scratch_dir),
        }


def build_run_layout(experiment_id: str, root: str | Path = "runs") -> RunLayout:
    run_root = Path(root)
    experiment_root = run_root / experiment_id
    control_dir = experiment_root / "control"
    artifacts_dir = experiment_root / "artifacts"
    scratch_dir = experiment_root / "scratch"
    return RunLayout(
        root=run_root,
        experiment_root=experiment_root,
        manifest_path=experiment_root / "manifest.json",
        heartbeat_path=experiment_root / "heartbeat.json",
        summary_path=experiment_root / "summary.json",
        console_log_path=experiment_root / "console.log",
        events_path=experiment_root / "events.ndjson",
        control_dir=control_dir,
        stop_flag_path=control_dir / "STOP",
        reload_flag_path=control_dir / "RELOAD",
        note_path=control_dir / "NOTE.txt",
        artifacts_dir=artifacts_dir,
        scratch_dir=scratch_dir,
    )
