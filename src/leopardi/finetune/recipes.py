from __future__ import annotations

from dataclasses import asdict

from leopardi.finetune.config import (
    AdapterConfig,
    FinetuneLossWeights,
    FinetuneOptimizerConfig,
    FinetuneRuntimeConfig,
    FinetuneStageConfig,
    RewardWeights,
)


def finetune_stage_recipe(stage: str) -> FinetuneStageConfig:
    runtime = FinetuneRuntimeConfig()

    recipes = {
        "f0_general_sft": FinetuneStageConfig(
            stage="f0_general_sft",
            visual_mode="standard",
            adapter=AdapterConfig(mode="full", rank=0, alpha=0, dropout=0.0),
            optimizer=FinetuneOptimizerConfig(lr=1e-4, weight_decay=0.05),
            runtime=runtime,
            loss_weights=FinetuneLossWeights(
                token_ce=1.0,
                repair_ce=0.0,
                block_type=0.15,
                block_length=0.05,
                specialist_hint=0.08,
                block_box=0.05,
                rotation=0.03,
                handwriting=0.03,
                formula_tokens=0.08,
                table_blocks=0.08,
                table_spans=0.04,
                reward_anchor=0.0,
            ),
            reward_weights=RewardWeights(),
        ),
        "f1_specialist_sft": FinetuneStageConfig(
            stage="f1_specialist_sft",
            visual_mode="hard",
            adapter=AdapterConfig(mode="full", rank=0, alpha=0, dropout=0.0),
            optimizer=FinetuneOptimizerConfig(lr=8e-5, weight_decay=0.05),
            runtime=runtime,
            loss_weights=FinetuneLossWeights(
                token_ce=1.0,
                repair_ce=0.15,
                block_type=0.2,
                block_length=0.05,
                specialist_hint=0.12,
                block_box=0.08,
                rotation=0.08,
                handwriting=0.08,
                formula_tokens=0.15,
                table_blocks=0.15,
                table_spans=0.08,
                reward_anchor=0.0,
            ),
            reward_weights=RewardWeights(),
        ),
        "f2_repair_sft": FinetuneStageConfig(
            stage="f2_repair_sft",
            track="s0-repair",
            visual_mode="hard",
            adapter=AdapterConfig(mode="full", rank=0, alpha=0, dropout=0.0),
            optimizer=FinetuneOptimizerConfig(lr=6e-5, weight_decay=0.03),
            runtime=runtime,
            loss_weights=FinetuneLossWeights(
                token_ce=0.6,
                repair_ce=1.0,
                block_type=0.1,
                block_length=0.05,
                specialist_hint=0.1,
                block_box=0.03,
                rotation=0.02,
                handwriting=0.02,
                formula_tokens=0.08,
                table_blocks=0.1,
                table_spans=0.05,
                reward_anchor=0.1,
            ),
            reward_weights=RewardWeights(),
        ),
        "f3_rlvr": FinetuneStageConfig(
            stage="f3_rlvr",
            visual_mode="hard",
            adapter=AdapterConfig(
                mode="lora",
                rank=32,
                alpha=64,
                dropout=0.05,
                target_modules=("writer", "planner", "latent_bottleneck"),
            ),
            optimizer=FinetuneOptimizerConfig(lr=5e-5, weight_decay=0.0),
            runtime=FinetuneRuntimeConfig(
                hardware_tag="rtx5090",
                precision="bf16",
                compile_model=False,
                gradient_checkpointing=True,
                micro_batch_size=1,
                gradient_accumulation_steps=8,
                dataloader_workers=2,
                max_steps=3000,
                log_every=10,
                eval_every=100,
                save_every=100,
            ),
            loss_weights=FinetuneLossWeights(
                token_ce=0.4,
                repair_ce=0.3,
                block_type=0.05,
                block_length=0.02,
                specialist_hint=0.05,
                block_box=0.02,
                rotation=0.0,
                handwriting=0.0,
                formula_tokens=0.03,
                table_blocks=0.03,
                table_spans=0.02,
                reward_anchor=1.0,
            ),
            reward_weights=RewardWeights(
                markdown_validity=1.2,
                latex_validity=1.0,
                table_validity=1.0,
                reading_order=0.5,
                edit_similarity=1.2,
                output_length_penalty=0.1,
                latency_penalty=0.25,
                repair_budget_penalty=0.2,
            ),
        ),
    }
    if stage not in recipes:
        raise KeyError(f"Unknown finetune stage recipe: {stage}")
    return recipes[stage]


def finetune_stage_recipe_dict(stage: str) -> dict[str, object]:
    return asdict(finetune_stage_recipe(stage))
