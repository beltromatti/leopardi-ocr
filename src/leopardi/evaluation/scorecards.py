from __future__ import annotations

from dataclasses import dataclass

from leopardi.evaluation.metrics import MetricCard, aggregate_metric_cards


@dataclass(slots=True)
class ScorecardRow:
    key: str
    values: dict[str, object]


def build_public_frontier_scorecard(
    *,
    model_name: str,
    protocol_version: str,
    size_band: str,
    evidence_grade: str,
    cards: list[MetricCard],
) -> ScorecardRow:
    metrics = aggregate_metric_cards(cards)
    return ScorecardRow(
        key=model_name,
        values={
            "model": model_name,
            "protocol_version": protocol_version,
            "size_band": size_band,
            "page_overall": metrics.get("page_overall", metrics.get("normalized_edit_similarity", 0.0)),
            "markdown_validity": metrics.get("markdown_validity", 0.0),
            "latex_exact_match": metrics.get("latex_exact_match", 0.0),
            "table_teds": metrics.get("table_teds", 0.0),
            "rotation_score": metrics.get("rotation_score", 0.0),
            "wild_page_score": metrics.get("wild_page_score", 0.0),
            "p50_latency_ms_per_page": metrics.get("p50_latency_ms_per_page", 0.0),
            "params_total_b": metrics.get("params_total_b", 0.0),
            "evidence_grade": evidence_grade,
        },
    )


def build_internal_promotion_scorecard(
    *,
    experiment_id: str,
    track: str,
    protocol_version: str,
    difficulty_tier: str,
    cards: list[MetricCard],
    decision: str,
) -> ScorecardRow:
    metrics = aggregate_metric_cards(cards)
    return ScorecardRow(
        key=experiment_id,
        values={
            "experiment_id": experiment_id,
            "track": track,
            "protocol_version": protocol_version,
            "internal_difficulty_tier": difficulty_tier,
            "parsing_quality": metrics.get("document_overall", metrics.get("normalized_edit_similarity", 0.0)),
            "formulas": metrics.get("latex_exact_match", 0.0),
            "tables": metrics.get("table_teds", 0.0),
            "handwriting": metrics.get("handwriting_score", 0.0),
            "graphics": metrics.get("wild_page_score", 0.0),
            "latency": metrics.get("p50_latency_ms_per_page", 0.0),
            "decision": decision,
        },
    )


def build_size_normalized_scorecard(
    *,
    model_name: str,
    size_band: str,
    cards: list[MetricCard],
    lus: float,
) -> ScorecardRow:
    metrics = aggregate_metric_cards(cards)
    return ScorecardRow(
        key=model_name,
        values={
            "model": model_name,
            "size_band": size_band,
            "primary_public_score": metrics.get("page_overall", metrics.get("normalized_edit_similarity", 0.0)),
            "latency": metrics.get("p50_latency_ms_per_page", 0.0),
            "footprint": metrics.get("vram_peak_gib", 0.0),
            "LUS": lus,
        },
    )


def build_failure_slice_scorecard(cards: list[MetricCard]) -> list[ScorecardRow]:
    slice_names = ("formulas", "merged_cell_tables", "handwriting", "rotation", "graphics", "document_assembly")
    metrics = aggregate_metric_cards(cards)
    mapping = {
        "formulas": metrics.get("latex_exact_match", 0.0),
        "merged_cell_tables": metrics.get("table_teds", 0.0),
        "handwriting": metrics.get("handwriting_score", 0.0),
        "rotation": metrics.get("rotation_score", 0.0),
        "graphics": metrics.get("wild_page_score", 0.0),
        "document_assembly": metrics.get("document_overall", metrics.get("normalized_edit_similarity", 0.0)),
    }
    return [ScorecardRow(key=name, values={"slice": name, "score": mapping[name]}) for name in slice_names]
