from __future__ import annotations

from dataclasses import asdict

from leopardi.pretraining.config import (
    CurriculumConfig,
    DataMixConfig,
    ModuleLrConfig,
    ObjectiveWeights,
    OptimizerConfig,
    PretrainStageConfig,
    RuntimeConfig,
    SchedulerConfig,
)


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
            scheduler=SchedulerConfig(name="cosine", warmup_ratio=0.04, min_lr_ratio=0.2),
            data_mix=DataMixConfig(
                exact_pairs=0.85,
                synthetic_exact=0.0,
                structural_aux=0.0,
                table_aux=0.05,
                formula_aux=0.10,
                long_tail_aux=0.0,
                weak_label_discount=0.0,
            ),
            curriculum=CurriculumConfig(
                clean_phase_steps=2_000,
                transition_phase_steps=6_000,
                hard_phase_steps=0,
                hard_example_boost=1.0,
                pathological_boost=1.0,
                refresh_failure_pool_every=0,
                keep_easy_fraction=1.0,
            ),
            module_lr=ModuleLrConfig(
                visual_tokenizer=0.0,
                latent_bottleneck=0.8,
                planner=1.0,
                writer=1.25,
                auxiliary_heads=0.5,
            ),
            objective_weights=ObjectiveWeights(
                token_ce=1.0,
                formula_ce=0.2,
                table_ce=0.1,
                block_type=0.0,
                block_length=0.0,
                specialist_hint=0.0,
                block_box=0.0,
                planner_confidence=0.0,
                rotation=0.0,
                handwriting=0.0,
                formula_tokens=0.0,
                table_blocks=0.0,
                table_spans=0.0,
                label_smoothing=0.02,
            ),
        ),
        "p2_multimodal_core": PretrainStageConfig(
            stage="p2_multimodal_core",
            text_only=False,
            visual_mode="standard",
            optimizer=optimizer,
            runtime=runtime,
            scheduler=SchedulerConfig(name="cosine", warmup_ratio=0.03, min_lr_ratio=0.12),
            data_mix=DataMixConfig(
                exact_pairs=0.48,
                synthetic_exact=0.18,
                structural_aux=0.10,
                table_aux=0.11,
                formula_aux=0.10,
                long_tail_aux=0.03,
                weak_label_discount=0.30,
            ),
            curriculum=CurriculumConfig(),
            module_lr=ModuleLrConfig(),
            objective_weights=ObjectiveWeights(
                formula_ce=0.18,
                table_ce=0.18,
                block_type=0.22,
                specialist_hint=0.12,
                rotation=0.03,
                handwriting=0.03,
                formula_tokens=0.12,
                table_blocks=0.12,
                table_spans=0.06,
                planner_confidence=0.05,
                label_smoothing=0.01,
            ),
        ),
        "p3_hard_curriculum": PretrainStageConfig(
            stage="p3_hard_curriculum",
            text_only=False,
            visual_mode="hard",
            optimizer=optimizer,
            runtime=runtime,
            scheduler=SchedulerConfig(name="cosine", warmup_ratio=0.02, min_lr_ratio=0.2),
            data_mix=DataMixConfig(
                exact_pairs=0.28,
                synthetic_exact=0.25,
                structural_aux=0.10,
                table_aux=0.12,
                formula_aux=0.13,
                long_tail_aux=0.12,
                weak_label_discount=0.30,
            ),
            curriculum=CurriculumConfig(
                clean_phase_steps=1_000,
                transition_phase_steps=4_000,
                hard_phase_steps=30_000,
                hard_example_boost=1.35,
                pathological_boost=1.65,
                refresh_failure_pool_every=500,
                keep_easy_fraction=0.12,
            ),
            module_lr=ModuleLrConfig(
                visual_tokenizer=0.6,
                latent_bottleneck=0.9,
                planner=1.2,
                writer=1.3,
                auxiliary_heads=1.15,
            ),
            objective_weights=ObjectiveWeights(
                token_ce=1.0,
                formula_ce=0.2,
                table_ce=0.2,
                block_type=0.25,
                block_length=0.1,
                specialist_hint=0.15,
                block_box=0.1,
                planner_confidence=0.08,
                rotation=0.14,
                handwriting=0.12,
                formula_tokens=0.18,
                table_blocks=0.18,
                table_spans=0.12,
                label_smoothing=0.01,
            ),
        ),
    }
    if stage not in recipes:
        raise KeyError(f"Unknown stage recipe: {stage}")
    return recipes[stage]


def stage_recipe_dict(stage: str) -> dict[str, object]:
    recipe = stage_recipe(stage)
    return asdict(recipe)
