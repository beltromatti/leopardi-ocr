from __future__ import annotations

from dataclasses import asdict

from leopardi.pretraining.config import ObjectiveWeights, OptimizerConfig, PretrainStageConfig, RuntimeConfig


def stage_recipe(stage: str) -> PretrainStageConfig:
    runtime = RuntimeConfig()
    optimizer = OptimizerConfig()

    recipes = {
        "p1_text_warmup": PretrainStageConfig(
            stage="p1_text_warmup",
            text_only=True,
            visual_mode="fast",
            optimizer=optimizer,
            runtime=runtime,
            objective_weights=ObjectiveWeights(
                token_ce=1.0,
                block_type=0.0,
                block_length=0.0,
                specialist_hint=0.0,
                block_box=0.0,
                rotation=0.0,
                handwriting=0.0,
                formula_tokens=0.0,
                table_blocks=0.0,
                table_spans=0.0,
            ),
        ),
        "p2_multimodal_core": PretrainStageConfig(
            stage="p2_multimodal_core",
            text_only=False,
            visual_mode="standard",
            optimizer=optimizer,
            runtime=runtime,
            objective_weights=ObjectiveWeights(),
        ),
        "p3_hard_curriculum": PretrainStageConfig(
            stage="p3_hard_curriculum",
            text_only=False,
            visual_mode="hard",
            optimizer=optimizer,
            runtime=runtime,
            objective_weights=ObjectiveWeights(
                token_ce=1.0,
                block_type=0.25,
                block_length=0.1,
                specialist_hint=0.15,
                block_box=0.1,
                rotation=0.1,
                handwriting=0.1,
                formula_tokens=0.15,
                table_blocks=0.15,
                table_spans=0.1,
            ),
        ),
    }
    if stage not in recipes:
        raise KeyError(f"Unknown stage recipe: {stage}")
    return recipes[stage]


def stage_recipe_dict(stage: str) -> dict[str, object]:
    recipe = stage_recipe(stage)
    return asdict(recipe)
