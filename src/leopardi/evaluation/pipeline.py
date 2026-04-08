from __future__ import annotations

from dataclasses import dataclass

from leopardi.evaluation.config import EvaluationStageConfig
from leopardi.evaluation.metrics import EvaluationSample, MetricCard, aggregate_metric_cards, evaluate_sample
from leopardi.evaluation.registry import BaselineRegistryEntry, DatasetRegistryEntry
from leopardi.evaluation.reports import ReportPackage, build_report_package, report_package_dict
from leopardi.evaluation.scorecards import (
    ScorecardRow,
    build_failure_slice_scorecard,
    build_internal_promotion_scorecard,
    build_public_frontier_scorecard,
    build_size_normalized_scorecard,
)


@dataclass(slots=True)
class EvaluationResultBundle:
    metric_cards: list[MetricCard]
    aggregate_metrics: dict[str, float]
    scorecards: list[ScorecardRow]
    failure_slices: list[ScorecardRow]
    report_package: ReportPackage


def _dataset_bundle_summary(
    *,
    stage: EvaluationStageConfig,
    datasets: list[DatasetRegistryEntry],
) -> dict[str, object]:
    covered = [entry for entry in datasets if stage.protocol in entry.covered_protocols]
    return {
        "bundle_id": stage.bundle_id,
        "protocol_version": stage.protocol,
        "dataset_families": [entry.dataset_family for entry in covered],
        "public_benchmarks": list(stage.public_benchmarks),
        "difficulty_tiers": list(stage.difficulty_tiers),
        "required_slices": list(stage.required_slices),
    }


def _competitor_comparison(
    *,
    baselines: list[BaselineRegistryEntry],
    protocol: str,
) -> list[dict[str, object]]:
    return [
        {
            "baseline_id": baseline.baseline_id,
            "model_name": baseline.model_name,
            "size_band": baseline.size_band,
            "open_status": baseline.open_status,
            "primary_role": baseline.primary_role,
            "evidence_policy": baseline.evidence_policy,
            "notes": baseline.notes,
        }
        for baseline in baselines
        if protocol in baseline.main_protocols
    ]


def compile_evaluation_result(
    *,
    experiment_id: str,
    stage: EvaluationStageConfig,
    runtime_family: str,
    decode_mode: str,
    model_name: str,
    size_band: str,
    evidence_grade: str,
    samples: list[EvaluationSample],
    datasets: list[DatasetRegistryEntry],
    baselines: list[BaselineRegistryEntry],
    params_total_b: float,
    lus: float = 0.0,
    track: str = "evaluation",
    decision: str = "candidate",
    artifact_pointers: list[dict[str, str]] | None = None,
    evidence_notes: list[str] | None = None,
) -> EvaluationResultBundle:
    cards = [evaluate_sample(sample) for sample in samples]
    for card in cards:
        card.metrics.setdefault("params_total_b", params_total_b)
    aggregate = aggregate_metric_cards(cards)
    aggregate["params_total_b"] = params_total_b

    scorecards: list[ScorecardRow] = []
    if stage.protocol.startswith("public_frontier"):
        scorecards.append(
            build_public_frontier_scorecard(
                model_name=model_name,
                protocol_version=stage.protocol,
                size_band=size_band,
                evidence_grade=evidence_grade,
                cards=cards,
            )
        )
    else:
        difficulty_tier = stage.difficulty_tiers[-1] if stage.difficulty_tiers else "unknown"
        scorecards.append(
            build_internal_promotion_scorecard(
                experiment_id=experiment_id,
                track=track,
                protocol_version=stage.protocol,
                difficulty_tier=difficulty_tier,
                cards=cards,
                decision=decision,
            )
        )
    scorecards.append(
        build_size_normalized_scorecard(
            model_name=model_name,
            size_band=size_band,
            cards=cards,
            lus=lus,
        )
    )

    failure_slices = build_failure_slice_scorecard(cards)
    report = build_report_package(
        protocol_version=stage.protocol,
        experiment_id=experiment_id,
        hardware_tag=stage.runtime.hardware_tag,
        decode_mode=decode_mode,
        runtime_family=runtime_family,
        structured_output_backend=stage.runtime.structured_output_backend,
        dataset_bundle_summary=_dataset_bundle_summary(stage=stage, datasets=datasets),
        scorecards=scorecards,
        failure_slice_summary=failure_slices,
        competitor_comparison=_competitor_comparison(baselines=baselines, protocol=stage.protocol),
        cards=cards,
        artifact_pointers=artifact_pointers,
        evidence_notes=evidence_notes,
    )
    return EvaluationResultBundle(
        metric_cards=cards,
        aggregate_metrics=aggregate,
        scorecards=scorecards,
        failure_slices=failure_slices,
        report_package=report,
    )


def evaluation_result_bundle_dict(bundle: EvaluationResultBundle) -> dict[str, object]:
    return {
        "metric_cards": [
            {"metrics": card.metrics, "metadata": card.metadata}
            for card in bundle.metric_cards
        ],
        "aggregate_metrics": bundle.aggregate_metrics,
        "scorecards": [row.values for row in bundle.scorecards],
        "failure_slices": [row.values for row in bundle.failure_slices],
        "report_package": report_package_dict(bundle.report_package),
    }
