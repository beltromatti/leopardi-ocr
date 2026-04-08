from __future__ import annotations

from dataclasses import dataclass
from math import inf

from leopardi.optimization.config import OptimizationGoalConfig


@dataclass(slots=True)
class VariantMeasurement:
    variant_id: str
    overall_score: float
    markdown_validity: float
    latex_validity: float
    table_score: float
    formula_score: float
    latency_ms: float
    peak_memory_gb: float
    throughput_pages_per_second: float


@dataclass(slots=True)
class RankedVariant:
    variant_id: str
    score: float
    quality_retention: float
    latency_gain: float
    memory_gain: float
    passes_floors: bool


def _passes_quality_floors(
    reference: VariantMeasurement,
    candidate: VariantMeasurement,
    goal: OptimizationGoalConfig,
) -> bool:
    if candidate.overall_score < goal.quality_floor:
        return False
    if candidate.markdown_validity < goal.markdown_validity_floor:
        return False
    if candidate.latex_validity < goal.latex_validity_floor:
        return False
    if candidate.peak_memory_gb > goal.max_memory_gb:
        return False
    quality_retention = candidate.overall_score / max(reference.overall_score, 1e-8)
    if 1.0 - quality_retention > goal.max_relative_quality_drop:
        return False
    latency_gain = max(reference.latency_ms, 1e-8) / max(candidate.latency_ms, 1e-8) - 1.0
    return latency_gain >= goal.min_latency_gain


def rank_candidates(
    reference: VariantMeasurement,
    candidates: list[VariantMeasurement],
    goal: OptimizationGoalConfig,
) -> list[RankedVariant]:
    ranked: list[RankedVariant] = []
    for candidate in candidates:
        quality_retention = candidate.overall_score / max(reference.overall_score, 1e-8)
        latency_gain = max(reference.latency_ms, 1e-8) / max(candidate.latency_ms, 1e-8)
        memory_gain = max(reference.peak_memory_gb, 1e-8) / max(candidate.peak_memory_gb, 1e-8)
        passes = _passes_quality_floors(reference, candidate, goal)
        score = -inf
        if passes:
            score = 0.55 * quality_retention + 0.3 * latency_gain + 0.15 * memory_gain
        ranked.append(
            RankedVariant(
                variant_id=candidate.variant_id,
                score=score,
                quality_retention=quality_retention,
                latency_gain=latency_gain,
                memory_gain=memory_gain,
                passes_floors=passes,
            )
        )
    return sorted(ranked, key=lambda item: item.score, reverse=True)


def pareto_frontier(candidates: list[VariantMeasurement]) -> list[VariantMeasurement]:
    frontier: list[VariantMeasurement] = []
    for candidate in candidates:
        dominated = False
        for other in candidates:
            if other.variant_id == candidate.variant_id:
                continue
            better_or_equal = (
                other.overall_score >= candidate.overall_score
                and other.latency_ms <= candidate.latency_ms
                and other.peak_memory_gb <= candidate.peak_memory_gb
            )
            strictly_better = (
                other.overall_score > candidate.overall_score
                or other.latency_ms < candidate.latency_ms
                or other.peak_memory_gb < candidate.peak_memory_gb
            )
            if better_or_equal and strictly_better:
                dominated = True
                break
        if not dominated:
            frontier.append(candidate)
    return sorted(frontier, key=lambda item: (-item.overall_score, item.latency_ms, item.peak_memory_gb))
