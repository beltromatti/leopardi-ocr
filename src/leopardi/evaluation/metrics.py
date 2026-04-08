from __future__ import annotations

from dataclasses import dataclass, field
from difflib import SequenceMatcher
from statistics import median

from leopardi.evaluation.normalization import normalize_latex, normalize_markdown
from leopardi.inference.config import ValidationConfig
from leopardi.inference.validation import validate_markdown
from leopardi.schemas.output import ParsedPage


@dataclass(slots=True)
class EvaluationSample:
    sample_id: str
    dataset_family: str
    decode_mode: str
    prediction_markdown: str
    reference_markdown: str
    latency_ms: float
    output_tokens: int
    difficulty: str | None = None
    prediction_page: ParsedPage | None = None
    reference_page: ParsedPage | None = None
    formula_prediction: str | None = None
    formula_reference: str | None = None
    native_metrics: dict[str, float] = field(default_factory=dict)
    failure_slices: tuple[str, ...] = ()
    gpu_type: str = "rtx5090"
    precision_mode: str = "bf16"
    batch_size: int = 1
    input_resolution_policy: str = "adaptive_1280"
    protocol_version: str = "public_frontier_v1"
    deployment_class: str = "single_gpu"


@dataclass(slots=True)
class MetricCard:
    metrics: dict[str, float]
    metadata: dict[str, str]


def _safe_similarity(a: str, b: str) -> float:
    if not a and not b:
        return 1.0
    return SequenceMatcher(a=a, b=b).ratio()


def markdown_validity_score(text: str) -> float:
    report = validate_markdown(text, _DEFAULT_VALIDATION)
    return 1.0 if report.valid else 0.0


def normalized_edit_similarity(prediction: str, reference: str) -> float:
    return _safe_similarity(normalize_markdown(prediction), normalize_markdown(reference))


def latex_exact_match(prediction: str | None, reference: str | None) -> float:
    if prediction is None or reference is None:
        return 0.0
    return 1.0 if normalize_latex(prediction) == normalize_latex(reference) else 0.0


def block_f1(prediction: ParsedPage | None, reference: ParsedPage | None) -> float:
    if prediction is None or reference is None:
        return 0.0
    pred_types = [block.type.value for block in prediction.blocks]
    ref_types = [block.type.value for block in reference.blocks]
    if not pred_types and not ref_types:
        return 1.0
    pred_set = set(pred_types)
    ref_set = set(ref_types)
    tp = len(pred_set & ref_set)
    if tp == 0:
        return 0.0
    precision = tp / max(len(pred_set), 1)
    recall = tp / max(len(ref_set), 1)
    return 2 * precision * recall / max(precision + recall, 1e-8)


def evaluate_sample(sample: EvaluationSample) -> MetricCard:
    metrics = {
        "normalized_edit_similarity": normalized_edit_similarity(
            sample.prediction_markdown,
            sample.reference_markdown,
        ),
        "markdown_validity": markdown_validity_score(sample.prediction_markdown),
        "block_f1": block_f1(sample.prediction_page, sample.reference_page),
        "latex_exact_match": latex_exact_match(sample.formula_prediction, sample.formula_reference),
        "p50_latency_ms_per_page": sample.latency_ms,
        "p95_latency_ms_per_page": sample.latency_ms,
        "output_tokens_per_page": float(sample.output_tokens),
    }
    metrics.update(sample.native_metrics)
    return MetricCard(
        metrics=metrics,
        metadata={
            "sample_id": sample.sample_id,
            "dataset_family": sample.dataset_family,
            "decode_mode": sample.decode_mode,
            "difficulty": sample.difficulty or "unknown",
            "gpu_type": sample.gpu_type,
            "precision_mode": sample.precision_mode,
            "batch_size": str(sample.batch_size),
            "input_resolution_policy": sample.input_resolution_policy,
            "protocol_version": sample.protocol_version,
            "deployment_class": sample.deployment_class,
        },
    )


def aggregate_metric_cards(cards: list[MetricCard]) -> dict[str, float]:
    if not cards:
        return {}
    metric_names = sorted({name for card in cards for name in card.metrics})
    aggregated: dict[str, float] = {}
    for name in metric_names:
        values = [card.metrics[name] for card in cards if name in card.metrics]
        if not values:
            continue
        if name == "p95_latency_ms_per_page":
            aggregated[name] = max(values)
        elif name == "p50_latency_ms_per_page":
            aggregated[name] = float(median(values))
        else:
            aggregated[name] = sum(values) / len(values)
    if "p50_latency_ms_per_page" in aggregated:
        aggregated["pages_per_second"] = 1000.0 / max(aggregated["p50_latency_ms_per_page"], 1e-8)
    return aggregated


_DEFAULT_VALIDATION = ValidationConfig()
