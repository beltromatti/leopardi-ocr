from __future__ import annotations

from dataclasses import asdict, dataclass

from leopardi.evaluation.metrics import MetricCard
from leopardi.evaluation.scorecards import ScorecardRow


@dataclass(slots=True)
class ReportPackage:
    protocol_version: str
    experiment_id: str
    hardware_tag: str
    decode_mode: str
    runtime_family: str
    structured_output_backend: str
    artifact_pointers: list[dict[str, str]]
    dataset_bundle_summary: dict[str, object]
    scorecards: list[dict[str, object]]
    latency_card: dict[str, float]
    failure_slice_summary: list[dict[str, object]]
    competitor_comparison: list[dict[str, object]]
    evidence_notes: list[str]


def report_package_dict(package: ReportPackage) -> dict[str, object]:
    return asdict(package)


def build_report_package(
    *,
    protocol_version: str,
    experiment_id: str,
    hardware_tag: str,
    decode_mode: str,
    runtime_family: str,
    structured_output_backend: str,
    dataset_bundle_summary: dict[str, object],
    scorecards: list[ScorecardRow],
    failure_slice_summary: list[ScorecardRow],
    competitor_comparison: list[dict[str, object]],
    cards: list[MetricCard],
    artifact_pointers: list[dict[str, str]] | None = None,
    evidence_notes: list[str] | None = None,
) -> ReportPackage:
    p50_values = [card.metrics.get("p50_latency_ms_per_page", 0.0) for card in cards]
    p95_values = [card.metrics.get("p95_latency_ms_per_page", 0.0) for card in cards]
    latency_card = {
        "p50_latency_ms_per_page": sum(p50_values) / len(p50_values) if p50_values else 0.0,
        "p95_latency_ms_per_page": max(p95_values) if p95_values else 0.0,
        "sample_count": float(len(cards)),
    }
    return ReportPackage(
        protocol_version=protocol_version,
        experiment_id=experiment_id,
        hardware_tag=hardware_tag,
        decode_mode=decode_mode,
        runtime_family=runtime_family,
        structured_output_backend=structured_output_backend,
        artifact_pointers=artifact_pointers or [],
        dataset_bundle_summary=dataset_bundle_summary,
        scorecards=[row.values for row in scorecards],
        latency_card=latency_card,
        failure_slice_summary=[row.values for row in failure_slice_summary],
        competitor_comparison=competitor_comparison,
        evidence_notes=evidence_notes or [],
    )
