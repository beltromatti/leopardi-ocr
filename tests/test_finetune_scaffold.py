from __future__ import annotations

from leopardi.finetune.batch import FinetuneBatch
from leopardi.finetune.config import FinetuneStageConfig
from leopardi.finetune.losses import compute_finetune_losses
from leopardi.finetune.rewards import compute_reward_breakdown
from leopardi.finetune.runtime import (
    build_finetune_optimizer,
    materialize_finetune_stage,
    optimizer_group_summary,
)
from leopardi.model import LeopardiS0, LeopardiS0Config


def test_finetune_smoke_loss_and_reward() -> None:
    config = LeopardiS0Config.tiny()
    model = LeopardiS0(config)
    stage = FinetuneStageConfig(stage="f0_general_sft", visual_mode="standard")
    batch = FinetuneBatch.synthetic(
        batch_size=1,
        image_size=(128, 128),
        seq_len=24,
        vocab_size=config.writer_decoder.vocab_size,
        planner_blocks=config.planner.num_blocks,
        visual_tokens=25,
        num_block_types=len(config.planner.block_types),
        num_length_buckets=config.planner.num_length_buckets,
        num_hints=len(config.planner.specialist_hints),
        rotation_classes=config.auxiliary_heads.rotation_classes,
        handwriting_classes=config.auxiliary_heads.handwriting_classes,
    )

    outputs = model(batch.image, batch.decoder_input_ids, visual_mode=stage.visual_mode)
    loss_report = compute_finetune_losses(outputs, batch, stage)
    reward_report = compute_reward_breakdown(batch.reward_signals or {}, stage)

    assert loss_report.total_loss.item() > 0.0
    assert "token_ce" in loss_report.loss_terms
    assert "formula_ce" in loss_report.loss_terms
    assert isinstance(reward_report.reward_terms, dict)
    assert "formula_exactness" in reward_report.reward_terms


def test_finetune_optimizer_groups_and_materialization(tmp_path) -> None:
    config = LeopardiS0Config.tiny()
    model = LeopardiS0(config)
    stage = FinetuneStageConfig(stage="f3_rlvr", visual_mode="hard")
    optimizer = build_finetune_optimizer(model, stage)

    assert len(optimizer.param_groups) >= 1
    assert optimizer_group_summary(model, stage)

    payload = materialize_finetune_stage(
        experiment_id="leo-s0-f3-test",
        stage=stage,
        model_config_path="configs/model/leopardi_s0.yaml",
        root=tmp_path / "runs",
    )
    assert payload["plan"]["stage"] == "f3_rlvr"
    assert (
        tmp_path / "runs" / "leo-s0-f3-test" / "artifacts" / "finetune" / "f3_rlvr" / "finetune-plan.json"
    ).exists()
