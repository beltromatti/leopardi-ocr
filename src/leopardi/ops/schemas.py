from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field


class ArtifactPointer(BaseModel):
    artifact_kind: Literal["checkpoint", "bundle", "report", "release_card", "summary_table", "runtime_plan", "request_template"]
    uri: str
    local_path: str | None = None
    checksum: str | None = None
    persistence_status: Literal["local_only", "queued", "published", "verified"] = "queued"


class RunHeartbeat(BaseModel):
    experiment_id: str
    phase: str
    stage: str
    state: Literal["draft", "running", "paused", "stopping", "completed", "failed"]
    current_step: int = 0
    latest_metrics: dict[str, float] = Field(default_factory=dict)
    last_save_step: int | None = None
    last_save_at: str | None = None
    last_sync_at: str | None = None
    last_sync_status: Literal["unknown", "ok", "degraded", "failed"] = "unknown"
    note: str | None = None


class RunManifest(BaseModel):
    experiment_id: str
    phase: Literal["data_pipeline", "pretraining", "finetune", "optimization", "inference", "evaluation", "serve"]
    stage: str
    track: str
    hardware_tag: str
    config_paths: list[str]
    data_bundle_ids: list[str] = Field(default_factory=list)
    protocol_version: str | None = None
    local_run_root: str
    persistent_targets: dict[str, str] = Field(default_factory=dict)


class RunSummary(BaseModel):
    experiment_id: str
    phase: str
    stage: str
    outcome: Literal["completed", "stopped", "failed"]
    key_metrics: dict[str, float] = Field(default_factory=dict)
    artifacts: list[ArtifactPointer] = Field(default_factory=list)
    notes: list[str] = Field(default_factory=list)
