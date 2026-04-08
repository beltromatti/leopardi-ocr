from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path


def _read_csv(path: str | Path) -> list[dict[str, str]]:
    with Path(path).open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def _split_pipe(value: str) -> tuple[str, ...]:
    return tuple(item for item in value.split("|") if item)


@dataclass(slots=True)
class SourceRegistryEntry:
    source_id: str
    source_name: str
    data_class: str
    primary_role: str
    task_coverage: tuple[str, ...]
    target_authority: str
    license_or_access_status: str
    ingestion_priority: str
    retention_mode: str
    notes: str


@dataclass(slots=True)
class SourceStatusEntry:
    source_id: str
    approved_role: str
    current_status: str
    next_action: str
    notes: str


@dataclass(slots=True)
class BundleRegistryEntry:
    bundle_id: str
    stage: str
    bundle_class: str
    primary_sources: tuple[str, ...]
    data_class_mix: str
    persistent_store: str
    target_status: str
    notes: str


@dataclass(slots=True)
class BuildProfileEntry:
    profile_id: str
    focus: str
    bundles: tuple[str, ...]
    source_groups: tuple[str, ...]
    use_case: str


@dataclass(slots=True)
class PublishRegistryEntry:
    artifact_group: str
    persistent_namespace: str
    artifact_kind: str
    required_for_release: bool
    git_companion_required: bool
    verification_rule: str
    retention_policy: str


def load_source_registry(
    path: str | Path = "data_pipeline/ingestion/source-registry.csv",
) -> list[SourceRegistryEntry]:
    rows = _read_csv(path)
    return [
        SourceRegistryEntry(
            source_id=row["source_id"],
            source_name=row["source_name"],
            data_class=row["data_class"],
            primary_role=row["primary_role"],
            task_coverage=_split_pipe(row["task_coverage"]),
            target_authority=row["target_authority"],
            license_or_access_status=row["license_or_access_status"],
            ingestion_priority=row["ingestion_priority"],
            retention_mode=row["retention_mode"],
            notes=row["notes"],
        )
        for row in rows
    ]


def load_source_status(
    path: str | Path = "data_pipeline/registry/source-status.csv",
) -> list[SourceStatusEntry]:
    rows = _read_csv(path)
    return [
        SourceStatusEntry(
            source_id=row["source_id"],
            approved_role=row["approved_role"],
            current_status=row["current_status"],
            next_action=row["next_action"],
            notes=row["notes"],
        )
        for row in rows
    ]


def load_bundle_registry(
    path: str | Path = "data_pipeline/registry/bundle-registry.csv",
) -> list[BundleRegistryEntry]:
    rows = _read_csv(path)
    return [
        BundleRegistryEntry(
            bundle_id=row["bundle_id"],
            stage=row["stage"],
            bundle_class=row["bundle_class"],
            primary_sources=_split_pipe(row["primary_sources"]),
            data_class_mix=row["data_class_mix"],
            persistent_store=row["persistent_store"],
            target_status=row["target_status"],
            notes=row["notes"],
        )
        for row in rows
    ]


def load_build_profiles(
    path: str | Path = "data_pipeline/profiles/profile-registry.csv",
) -> list[BuildProfileEntry]:
    rows = _read_csv(path)
    return [
        BuildProfileEntry(
            profile_id=row["profile_id"],
            focus=row["focus"],
            bundles=_split_pipe(row["bundles"]),
            source_groups=_split_pipe(row["source_groups"]),
            use_case=row["use_case"],
        )
        for row in rows
    ]


def load_publish_registry(
    path: str | Path = "data_pipeline/registry/publish-registry.csv",
) -> list[PublishRegistryEntry]:
    rows = _read_csv(path)
    return [
        PublishRegistryEntry(
            artifact_group=row["artifact_group"],
            persistent_namespace=row["persistent_namespace"],
            artifact_kind=row["artifact_kind"],
            required_for_release=row["required_for_release"] == "yes",
            git_companion_required=row["git_companion_required"] == "yes",
            verification_rule=row["verification_rule"],
            retention_policy=row["retention_policy"],
        )
        for row in rows
    ]


def registry_summary() -> dict[str, object]:
    sources = load_source_registry()
    bundles = load_bundle_registry()
    profiles = load_build_profiles()
    publish = load_publish_registry()
    statuses = load_source_status()
    return {
        "source_count": len(sources),
        "approved_source_count": sum(item.current_status == "approved" for item in statuses),
        "bundle_count": len(bundles),
        "profile_count": len(profiles),
        "publish_groups": sorted(item.artifact_group for item in publish),
        "exact_foundations": sorted(
            item.source_id for item in sources if item.data_class == "exact_pair"
        ),
    }
