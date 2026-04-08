from __future__ import annotations

from dataclasses import asdict, dataclass

from leopardi.inference.config import InferenceStageConfig


@dataclass(slots=True)
class InferenceArtifactCard:
    experiment_id: str
    stage: str
    track: str
    artifact_variant_id: str
    artifact_uri: str
    runtime_family: str
    fallback_runtime_family: str
    structured_backend_default: str
    target_longest_image_dim: int
    max_page_pixels: int
    primary_report_uri: str
    metadata_uri: str
    modes: tuple[dict[str, object], ...]


def build_inference_artifact_card(
    *,
    experiment_id: str,
    stage: InferenceStageConfig,
    primary_report_uri: str,
    metadata_uri: str,
    mode_summaries: tuple[dict[str, object], ...],
) -> InferenceArtifactCard:
    return InferenceArtifactCard(
        experiment_id=experiment_id,
        stage=stage.stage,
        track=stage.track,
        artifact_variant_id=stage.artifact_variant_id,
        artifact_uri=stage.artifact_uri,
        runtime_family=stage.runtime_family,
        fallback_runtime_family=stage.fallback_runtime_family,
        structured_backend_default=stage.structured_backend_default,
        target_longest_image_dim=stage.render.target_longest_image_dim,
        max_page_pixels=stage.render.max_page_pixels,
        primary_report_uri=primary_report_uri,
        metadata_uri=metadata_uri,
        modes=mode_summaries,
    )


def inference_artifact_card_dict(card: InferenceArtifactCard) -> dict[str, object]:
    return asdict(card)
