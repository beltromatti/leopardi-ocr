from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml


@dataclass(slots=True)
class DataBuildPersistenceConfig:
    bundle_target: str = "hf://leopardi-ocr-data-bundles"
    metadata_target: str = "hf://leopardi-ocr-data-metadata"
    upload_mode: str = "upload_large_folder"
    upload_cache_dir: str = ".cache/huggingface"
    sync_policy: str = "bundle_boundary_and_final"


@dataclass(slots=True)
class DataBuildRuntimeConfig:
    hardware_tag: str = "rtx5090"
    cpu_workers: int = 12
    io_workers: int = 16
    local_disk_budget_gb: int = 900
    max_active_sources: int = 2
    priority: tuple[str, ...] = ("resumability", "bounded_disk_usage", "publish_then_purge")
    implementation_notes: tuple[str, ...] = ()
    persistence: DataBuildPersistenceConfig = field(default_factory=DataBuildPersistenceConfig)


@dataclass(slots=True)
class DataBuildStageConfig:
    stage: str
    profile_id: str
    target_model_family: str = "leopardi_s0"
    target_param_budget_m: int = 100
    future_scale_family: str = "leopardi_s1"
    upstream_bundle_target: str | None = None
    failure_manifest_uri: str | None = None
    bundle_ids: tuple[str, ...] = ()
    source_ids: tuple[str, ...] = ()
    allow_research_watchlist: bool = False
    strict_disk_guard: bool = True
    shard_format: str = "webdataset"
    shard_target_size_mb: int = 256
    manifest_format: str = "parquet"
    raw_retention_mode: str = "publish_canonical_then_purge_raw"
    publish_verify_required: bool = True
    runtime: DataBuildRuntimeConfig = field(default_factory=DataBuildRuntimeConfig)

    @classmethod
    def from_dict(
        cls,
        payload: dict[str, Any],
        runtime_payload: dict[str, Any] | None = None,
    ) -> "DataBuildStageConfig":
        stage_payload = payload.get("data_build", payload)
        runtime_root = (runtime_payload or {}).get("runtime", runtime_payload or {})
        persistence_root = runtime_root.get("persistence", {})
        return cls(
            stage=stage_payload["stage"],
            profile_id=stage_payload["profile_id"],
            target_model_family=stage_payload.get("target_model_family", "leopardi_s0"),
            target_param_budget_m=stage_payload.get("target_param_budget_m", 100),
            future_scale_family=stage_payload.get("future_scale_family", "leopardi_s1"),
            upstream_bundle_target=stage_payload.get("upstream_bundle_target"),
            failure_manifest_uri=stage_payload.get("failure_manifest_uri"),
            bundle_ids=tuple(stage_payload.get("bundle_ids", ())),
            source_ids=tuple(stage_payload.get("source_ids", ())),
            allow_research_watchlist=stage_payload.get("allow_research_watchlist", False),
            strict_disk_guard=stage_payload.get("strict_disk_guard", True),
            shard_format=stage_payload.get("shard_format", "webdataset"),
            shard_target_size_mb=stage_payload.get("shard_target_size_mb", 256),
            manifest_format=stage_payload.get("manifest_format", "parquet"),
            raw_retention_mode=stage_payload.get(
                "raw_retention_mode", "publish_canonical_then_purge_raw"
            ),
            publish_verify_required=stage_payload.get("publish_verify_required", True),
            runtime=DataBuildRuntimeConfig(
                hardware_tag=runtime_root.get("hardware_tag", "rtx5090"),
                cpu_workers=runtime_root.get("cpu_workers", 12),
                io_workers=runtime_root.get("io_workers", 16),
                local_disk_budget_gb=runtime_root.get("local_disk_budget_gb", 900),
                max_active_sources=runtime_root.get("max_active_sources", 2),
                priority=tuple(
                    runtime_root.get(
                        "priority",
                        ("resumability", "bounded_disk_usage", "publish_then_purge"),
                    )
                ),
                implementation_notes=tuple(runtime_root.get("implementation_notes", ())),
                persistence=DataBuildPersistenceConfig(
                    bundle_target=persistence_root.get(
                        "bundle_target", "hf://leopardi-ocr-data-bundles"
                    ),
                    metadata_target=persistence_root.get(
                        "metadata_target", "hf://leopardi-ocr-data-metadata"
                    ),
                    upload_mode=persistence_root.get("upload_mode", "upload_large_folder"),
                    upload_cache_dir=persistence_root.get(
                        "upload_cache_dir", ".cache/huggingface"
                    ),
                    sync_policy=persistence_root.get(
                        "sync_policy", "bundle_boundary_and_final"
                    ),
                ),
            ),
        )

    @classmethod
    def from_yaml(
        cls,
        stage_path: str | Path,
        runtime_path: str | Path | None = None,
    ) -> "DataBuildStageConfig":
        with Path(stage_path).open("r", encoding="utf-8") as handle:
            stage_payload = yaml.safe_load(handle)
        runtime_payload: dict[str, Any] | None = None
        if runtime_path is not None:
            with Path(runtime_path).open("r", encoding="utf-8") as handle:
                runtime_payload = yaml.safe_load(handle)
        return cls.from_dict(stage_payload, runtime_payload=runtime_payload)
