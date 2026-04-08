from __future__ import annotations

from dataclasses import dataclass

from leopardi.data_pipeline.registry import (
    BuildProfileEntry,
    BundleRegistryEntry,
    load_build_profiles,
    load_bundle_registry,
    load_source_endpoints,
    load_source_registry,
    load_source_status,
)


@dataclass(slots=True)
class AuditFinding:
    severity: str
    code: str
    message: str


@dataclass(slots=True)
class AuditReport:
    ok: bool
    finding_count: int
    findings: tuple[AuditFinding, ...]
    summary: dict[str, object]


def _derived_sources_from_bundle(bundle: BundleRegistryEntry) -> tuple[str, ...]:
    return bundle.primary_sources


def _profile_source_resolution_is_possible(
    profile: BuildProfileEntry,
    source_ids: set[str],
    bundles: list[BundleRegistryEntry],
) -> bool:
    if all(source_id in source_ids for source_id in profile.source_groups):
        return True
    bundle_ids = set(profile.bundles)
    derivable_sources = {
        source_id
        for bundle in bundles
        if bundle.bundle_id in bundle_ids
        for source_id in _derived_sources_from_bundle(bundle)
    }
    return all(source_id in source_ids or source_id in derivable_sources for source_id in profile.source_groups)


def audit_data_pipeline() -> AuditReport:
    findings: list[AuditFinding] = []
    sources = load_source_registry()
    statuses = load_source_status()
    endpoints = load_source_endpoints()
    bundles = load_bundle_registry()
    profiles = load_build_profiles()

    source_ids = {entry.source_id for entry in sources}
    status_ids = {entry.source_id for entry in statuses}
    endpoint_ids = {entry.source_id for entry in endpoints}
    bundle_ids = {entry.bundle_id for entry in bundles}

    missing_status = sorted(source_ids - status_ids)
    for source_id in missing_status:
        findings.append(
            AuditFinding("error", "missing_source_status", f"Source `{source_id}` is missing status tracking.")
        )

    missing_endpoint = sorted(source_ids - endpoint_ids)
    for source_id in missing_endpoint:
        findings.append(
            AuditFinding(
                "error",
                "missing_source_endpoint",
                f"Source `{source_id}` is missing endpoint/probe policy coverage.",
            )
        )

    orphan_status = sorted(status_ids - source_ids)
    for source_id in orphan_status:
        findings.append(
            AuditFinding(
                "error",
                "orphan_source_status",
                f"Status row `{source_id}` does not exist in source registry.",
            )
        )

    orphan_endpoint = sorted(endpoint_ids - source_ids)
    for source_id in orphan_endpoint:
        findings.append(
            AuditFinding(
                "error",
                "orphan_source_endpoint",
                f"Endpoint row `{source_id}` does not exist in source registry.",
            )
        )

    endpoint_map = {entry.source_id: entry for entry in endpoints}
    for endpoint in endpoints:
        if endpoint.probe_policy == "automated":
            if not endpoint.probe_url or not endpoint.probe_method:
                findings.append(
                    AuditFinding(
                        "error",
                        "invalid_automated_probe",
                        f"Automated endpoint for `{endpoint.source_id}` needs method and URL.",
                    )
                )
        if endpoint.probe_policy == "internal" and endpoint.remote_fetchable:
            findings.append(
                AuditFinding(
                    "error",
                    "invalid_probe_policy",
                    f"Endpoint `{endpoint.source_id}` cannot be remote_fetchable with probe policy `{endpoint.probe_policy}`.",
                )
            )

    for bundle in bundles:
        for source_id in bundle.primary_sources:
            if source_id not in source_ids:
                findings.append(
                    AuditFinding(
                        "error",
                        "unknown_bundle_source",
                        f"Bundle `{bundle.bundle_id}` references unknown source `{source_id}`.",
                    )
                )

    for profile in profiles:
        for bundle_id in profile.bundles:
            if bundle_id not in bundle_ids:
                findings.append(
                    AuditFinding(
                        "error",
                        "unknown_profile_bundle",
                        f"Profile `{profile.profile_id}` references unknown bundle `{bundle_id}`.",
                    )
                )
        if not _profile_source_resolution_is_possible(profile, source_ids, bundles):
            findings.append(
                AuditFinding(
                    "error",
                    "unresolvable_profile_sources",
                    f"Profile `{profile.profile_id}` cannot resolve all source groups against source and bundle registries.",
                )
            )

    for source in sources:
        endpoint = endpoint_map.get(source.source_id)
        if endpoint is None:
            continue
        if source.retention_mode.startswith("publish_") and endpoint.probe_policy == "internal":
            continue
        if source.data_class in {"exact_pair", "trusted_aux"} and endpoint.probe_policy == "manual":
            findings.append(
                AuditFinding(
                    "warning",
                    "manual_fetch_path",
                    f"Source `{source.source_id}` requires a manual acquisition path; pin a concrete access route before first full build.",
                )
            )

    summary = {
        "source_count": len(sources),
        "bundle_count": len(bundles),
        "profile_count": len(profiles),
        "automated_probe_count": sum(entry.probe_policy == "automated" for entry in endpoints),
        "manual_probe_count": sum(entry.probe_policy == "manual" for entry in endpoints),
        "internal_probe_count": sum(entry.probe_policy == "internal" for entry in endpoints),
        "remote_fetchable_count": sum(entry.remote_fetchable for entry in endpoints),
        "error_count": sum(item.severity == "error" for item in findings),
        "warning_count": sum(item.severity == "warning" for item in findings),
    }
    return AuditReport(
        ok=summary["error_count"] == 0,
        finding_count=len(findings),
        findings=tuple(findings),
        summary=summary,
    )
