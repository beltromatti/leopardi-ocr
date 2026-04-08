from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path

from leopardi.data_pipeline.config import DataBuildStageConfig
from leopardi.data_pipeline.registry import (
    BuildProfileEntry,
    BundleRegistryEntry,
    PublishRegistryEntry,
    SourceRegistryEntry,
    SourceStatusEntry,
    load_build_profiles,
    load_bundle_registry,
    load_source_endpoints,
    load_publish_registry,
    load_source_registry,
    load_source_status,
)
from leopardi.ops import ensure_run_layout


@dataclass(slots=True)
class SourceWave:
    wave_index: int
    source_ids: tuple[str, ...]
    rationale: str


@dataclass(slots=True)
class BundleBuildSpec:
    bundle_id: str
    stage: str
    bundle_class: str
    source_ids: tuple[str, ...]
    local_staging_dir: str
    local_manifest_dir: str
    sample_artifact_group: str
    sample_uri: str
    bundle_uri: str
    manifest_uri: str
    retention_mode: str


@dataclass(slots=True)
class LocalPathPlan:
    metadata_cache_dir: str
    raw_cache_dir: str
    work_cache_dir: str
    upload_staging_dir: str
    publish_ledger_path: str


@dataclass(slots=True)
class DataBuildExecutionPlan:
    stage: str
    profile_id: str
    target_model_family: str
    target_param_budget_m: int
    source_ids: tuple[str, ...]
    bundle_ids: tuple[str, ...]
    source_waves: tuple[SourceWave, ...]
    local_paths: LocalPathPlan
    shard_format: str
    shard_target_size_mb: int
    manifest_format: str
    upload_mode: str
    strict_disk_guard: bool
    plan_path: str
    report_stub_path: str
    bundle_specs: tuple[BundleBuildSpec, ...]


def _priority_rank(status: SourceStatusEntry, source: SourceRegistryEntry) -> tuple[int, int, str]:
    next_action_rank = {
        "prioritize_build": 0,
        "ingest_after_exact_core": 1,
        "ingest_for_hardcases": 2,
        "verify_license": 3,
        "verify_release_terms": 4,
    }
    source_rank = {"A": 0, "B": 1, "C": 2}
    return (
        next_action_rank.get(status.next_action, 9),
        source_rank.get(source.ingestion_priority, 9),
        source.source_id,
    )


def _chunk(values: tuple[str, ...], size: int) -> tuple[tuple[str, ...], ...]:
    if size <= 0:
        return (values,)
    return tuple(values[index : index + size] for index in range(0, len(values), size))


def _publish_namespace_map(
    publish_entries: list[PublishRegistryEntry],
) -> dict[str, PublishRegistryEntry]:
    return {entry.artifact_group: entry for entry in publish_entries}


def _sample_artifact_group(bundle: BundleRegistryEntry) -> str:
    if bundle.stage == "holdout":
        return "holdouts"
    if "aux" in bundle.bundle_class or "trusted_aux" in bundle.data_class_mix:
        return "aux_samples"
    if "synthetic" in bundle.data_class_mix or bundle.bundle_id.startswith(("p3_", "f2_")):
        return "synthetic_samples"
    return "exact_samples"


def _resolve_profile(
    profile_id: str,
    profiles: list[BuildProfileEntry],
) -> BuildProfileEntry:
    for profile in profiles:
        if profile.profile_id == profile_id:
            return profile
    raise ValueError(f"Unknown data build profile: {profile_id}")


def _resolve_selected_sources(
    stage: DataBuildStageConfig,
    profile: BuildProfileEntry,
    sources: list[SourceRegistryEntry],
    statuses: list[SourceStatusEntry],
    bundles: list[BundleRegistryEntry],
) -> tuple[str, ...]:
    source_index = {entry.source_id: entry for entry in sources}
    status_index = {entry.source_id: entry for entry in statuses}
    requested = stage.source_ids or profile.source_groups
    if not stage.source_ids:
        unresolved = [source_id for source_id in requested if source_id not in source_index]
        if unresolved:
            bundle_ids = set(profile.bundles)
            derived = {
                source_id
                for bundle in bundles
                if bundle.bundle_id in bundle_ids
                for source_id in bundle.primary_sources
            }
            requested = tuple(
                source_id for source_id in dict.fromkeys((*requested, *sorted(derived))) if source_id in source_index
            )
    selected = []
    for source_id in requested:
        source = source_index.get(source_id)
        if source is None:
            continue
        if source.data_class == "weak_aux" and not stage.allow_research_watchlist:
            continue
        status = status_index.get(source_id)
        if status is None:
            continue
        if status.current_status != "approved" and not stage.allow_research_watchlist:
            continue
        selected.append(source_id)

    ranked = sorted(
        selected,
        key=lambda source_id: _priority_rank(status_index[source_id], source_index[source_id]),
    )
    return tuple(ranked)


def _resolve_selected_bundles(
    stage: DataBuildStageConfig,
    profile: BuildProfileEntry,
    bundles: list[BundleRegistryEntry],
) -> tuple[BundleRegistryEntry, ...]:
    bundle_index = {entry.bundle_id: entry for entry in bundles}
    requested = stage.bundle_ids or profile.bundles
    selected: list[BundleRegistryEntry] = []
    for bundle_id in requested:
        bundle = bundle_index.get(bundle_id)
        if bundle is not None:
            selected.append(bundle)
    return tuple(selected)


def build_data_build_execution_plan(
    *,
    experiment_id: str,
    stage: DataBuildStageConfig,
    stage_config_path: str,
    runtime_config_path: str,
    root: str | Path = "runs",
) -> DataBuildExecutionPlan:
    layout = ensure_run_layout(experiment_id, root=root)
    profiles = load_build_profiles()
    sources = load_source_registry()
    statuses = load_source_status()
    bundles = load_bundle_registry()
    endpoints = load_source_endpoints()
    publish_entries = load_publish_registry()

    profile = _resolve_profile(stage.profile_id, profiles)
    selected_sources = _resolve_selected_sources(stage, profile, sources, statuses, bundles)
    selected_bundles = _resolve_selected_bundles(stage, profile, bundles)
    endpoint_index = {entry.source_id: entry for entry in endpoints}
    fetchable_sources = tuple(
        source_id
        for source_id in selected_sources
        if endpoint_index.get(source_id) is not None and endpoint_index[source_id].remote_fetchable
    )
    waves = tuple(
        SourceWave(
            wave_index=index + 1,
            source_ids=wave,
            rationale="metadata-first selective acquisition within disk and bandwidth guardrails",
        )
        for index, wave in enumerate(_chunk(fetchable_sources, stage.runtime.max_active_sources))
    )

    stage_root = layout.artifacts_dir / "data_pipeline" / stage.stage
    scratch_root = layout.scratch_dir / "data_pipeline" / stage.stage
    local_paths = LocalPathPlan(
        metadata_cache_dir=str(scratch_root / "metadata-cache"),
        raw_cache_dir=str(scratch_root / "raw-cache"),
        work_cache_dir=str(scratch_root / "work-cache"),
        upload_staging_dir=str(scratch_root / "upload-staging"),
        publish_ledger_path=str(stage_root / "publish-ledger.stub.json"),
    )

    publish_map = _publish_namespace_map(publish_entries)
    source_set = set(selected_sources)
    bundle_specs: list[BundleBuildSpec] = []
    for bundle in selected_bundles:
        referenced_sources = tuple(
            source_id for source_id in bundle.primary_sources if source_id in source_set
        )
        if not referenced_sources:
            referenced_sources = selected_sources
        sample_group = _sample_artifact_group(bundle)
        sample_namespace = publish_map[sample_group].persistent_namespace
        bundle_specs.append(
            BundleBuildSpec(
                bundle_id=bundle.bundle_id,
                stage=bundle.stage,
                bundle_class=bundle.bundle_class,
                source_ids=referenced_sources,
                local_staging_dir=str(scratch_root / "bundles" / bundle.bundle_id),
                local_manifest_dir=str(stage_root / "manifests" / bundle.bundle_id),
                sample_artifact_group=sample_group,
                sample_uri=f"hf://{sample_namespace}/{bundle.bundle_id}",
                bundle_uri=f"{stage.runtime.persistence.bundle_target.rstrip('/')}/{bundle.bundle_id}",
                manifest_uri=f"{stage.runtime.persistence.metadata_target.rstrip('/')}/{bundle.bundle_id}",
                retention_mode=stage.raw_retention_mode,
            )
        )

    return DataBuildExecutionPlan(
        stage=stage.stage,
        profile_id=stage.profile_id,
        target_model_family=stage.target_model_family,
        target_param_budget_m=stage.target_param_budget_m,
        source_ids=selected_sources,
        bundle_ids=tuple(bundle.bundle_id for bundle in selected_bundles),
        source_waves=waves,
        local_paths=local_paths,
        shard_format=stage.shard_format,
        shard_target_size_mb=stage.shard_target_size_mb,
        manifest_format=stage.manifest_format,
        upload_mode=stage.runtime.persistence.upload_mode,
        strict_disk_guard=stage.strict_disk_guard,
        plan_path=str(stage_root / "data-build-plan.json"),
        report_stub_path=str(stage_root / "report.stub.json"),
        bundle_specs=tuple(bundle_specs),
    )


def plan_dict(plan: DataBuildExecutionPlan) -> dict[str, object]:
    payload = asdict(plan)
    payload["source_waves"] = [asdict(wave) for wave in plan.source_waves]
    payload["bundle_specs"] = [asdict(spec) for spec in plan.bundle_specs]
    payload["local_paths"] = asdict(plan.local_paths)
    return payload
