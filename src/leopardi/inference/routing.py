from __future__ import annotations

from dataclasses import dataclass

from leopardi.inference.config import DecodeModeConfig, InferenceStageConfig


@dataclass(slots=True)
class PageSignals:
    visual_density: float
    block_count_estimate: int
    formula_density: float
    table_density: float
    handwriting_likelihood: float
    chart_likelihood: float
    long_tiny_text_likelihood: float
    photo_distortion_likelihood: float
    orientation_uncertainty: float


@dataclass(slots=True)
class RoutingDecision:
    mode: str
    complexity_score: float
    reasons: tuple[str, ...]
    specialist_hints: tuple[str, ...]
    repair_budget: int
    visual_token_budget: int


def _clamp01(value: float) -> float:
    return max(0.0, min(1.0, value))


def _block_count_score(block_count_estimate: int) -> float:
    return _clamp01(block_count_estimate / 24.0)


def estimate_complexity_score(stage: InferenceStageConfig, signals: PageSignals) -> float:
    routing = stage.routing
    score = (
        routing.visual_density_weight * _clamp01(signals.visual_density)
        + routing.block_count_weight * _block_count_score(signals.block_count_estimate)
        + routing.formula_density_weight * _clamp01(signals.formula_density)
        + routing.table_density_weight * _clamp01(signals.table_density)
        + routing.handwriting_weight * _clamp01(signals.handwriting_likelihood)
        + routing.chart_weight * _clamp01(signals.chart_likelihood)
        + routing.long_tiny_text_weight * _clamp01(signals.long_tiny_text_likelihood)
        + routing.photo_distortion_weight * _clamp01(signals.photo_distortion_likelihood)
    )
    return _clamp01(score)


def route_page(stage: InferenceStageConfig, signals: PageSignals) -> RoutingDecision:
    routing = stage.routing
    score = estimate_complexity_score(stage, signals)
    reasons: list[str] = []
    specialist_hints: set[str] = set()
    mode_name = "fast"

    if signals.formula_density >= routing.hard_formula_density:
        reasons.append("formula_density")
        specialist_hints.add("formula")
    if signals.table_density >= routing.hard_table_density:
        reasons.append("table_density")
        specialist_hints.add("table")
    if signals.handwriting_likelihood >= routing.hard_handwriting_likelihood:
        reasons.append("handwriting")
        specialist_hints.add("handwriting")
    if signals.chart_likelihood >= routing.hard_chart_likelihood:
        reasons.append("chart")
        specialist_hints.add("chart")
    if signals.long_tiny_text_likelihood >= routing.hard_long_tiny_text_likelihood:
        reasons.append("long_tiny_text")
    if signals.photo_distortion_likelihood >= routing.hard_photo_distortion_likelihood:
        reasons.append("photo_distortion")
    if signals.orientation_uncertainty >= routing.hard_orientation_uncertainty:
        reasons.append("orientation_uncertainty")

    if reasons or score >= routing.hard_threshold:
        mode_name = "hard"
    elif score >= routing.standard_threshold:
        mode_name = "standard"
    else:
        mode_name = "fast"

    mode = stage.mode(mode_name)
    specialist_hints.update(mode.specialist_hints)
    return RoutingDecision(
        mode=mode.name,
        complexity_score=score,
        reasons=tuple(reasons) if reasons else ("low_complexity",),
        specialist_hints=tuple(sorted(specialist_hints)),
        repair_budget=mode.repair_budget,
        visual_token_budget=mode.visual_token_budget,
    )


def mode_summary(mode: DecodeModeConfig) -> dict[str, object]:
    return {
        "name": mode.name,
        "visual_token_budget": mode.visual_token_budget,
        "max_output_tokens": mode.max_output_tokens,
        "crop_budget": mode.crop_budget,
        "repair_budget": mode.repair_budget,
        "grammar_mode": mode.grammar_mode,
        "structured_output_backend": mode.structured_output_backend,
        "allow_block_local_repair": mode.allow_block_local_repair,
        "specialist_hints": list(mode.specialist_hints),
    }
