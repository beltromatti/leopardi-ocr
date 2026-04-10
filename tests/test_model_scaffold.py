from __future__ import annotations

from leopardi.model import LeopardiS0, LeopardiS0Config
from leopardi.pretraining.batch import PretrainBatch
from leopardi.pretraining.config import PretrainStageConfig
from leopardi.pretraining.losses import compute_pretraining_losses
from leopardi.pretraining.recipes import stage_recipe
from leopardi.pretraining.runtime import build_optimizer, materialize_pretraining_stage, optimizer_group_summary


def test_leopardi_s0_tiny_forward_shapes() -> None:
    config = LeopardiS0Config.tiny()
    model = LeopardiS0(config)
    batch = PretrainBatch.synthetic(
        batch_size=2,
        image_size=(128, 128),
        seq_len=32,
        vocab_size=config.writer_decoder.vocab_size,
        planner_blocks=config.planner.num_blocks,
        visual_tokens=16,
        num_block_types=len(config.planner.block_types),
        num_length_buckets=config.planner.num_length_buckets,
        num_hints=len(config.planner.specialist_hints),
        rotation_classes=config.auxiliary_heads.rotation_classes,
        handwriting_classes=config.auxiliary_heads.handwriting_classes,
    )

    outputs = model(batch.image, batch.decoder_input_ids, visual_mode="standard")

    assert outputs.decoder_logits.shape == (
        2,
        32,
        config.writer_decoder.vocab_size,
    )
    assert outputs.planner.block_type_logits.shape[:2] == (2, config.planner.num_blocks)
    assert outputs.visual_tokens.shape[1] > 0
    assert outputs.layout_tokens.shape[1] == 4
    assert outputs.mtp_logits is not None
    assert len(outputs.mtp_logits) == config.multi_token_prediction.horizon
    assert model.num_parameters() > 0


def test_canonical_model_scale_configs_are_aligned() -> None:
    s0 = LeopardiS0Config.from_yaml("configs/model/leopardi_s0.yaml")
    s1 = LeopardiS0Config.from_yaml("configs/model/leopardi_s1.yaml")

    assert s0.target_params_m == 200
    assert s0.hidden_size == 576
    assert s0.writer_decoder.num_layers == 12
    assert s0.writer_decoder.num_heads == 9
    assert s0.writer_decoder.num_kv_heads == 3
    assert s0.writer_decoder.pretrained_init == "HuggingFaceTB/SmolLM2-135M"

    assert s1.target_params_m == 600
    assert s1.hidden_size == 960
    assert s1.writer_decoder.num_layers == 27
    assert s1.writer_decoder.num_heads == 15
    assert s1.writer_decoder.num_kv_heads == 5
    assert s1.writer_decoder.pretrained_init == "HuggingFaceTB/SmolLM2-360M"


def test_pretraining_loss_report_smoke() -> None:
    config = LeopardiS0Config.tiny()
    model = LeopardiS0(config)
    stage = PretrainStageConfig(
        stage="p2_multimodal_core",
        visual_mode="standard",
    )
    batch = PretrainBatch.synthetic(
        batch_size=1,
        image_size=(128, 128),
        seq_len=24,
        vocab_size=config.writer_decoder.vocab_size,
        planner_blocks=config.planner.num_blocks,
        visual_tokens=16,
        num_block_types=len(config.planner.block_types),
        num_length_buckets=config.planner.num_length_buckets,
        num_hints=len(config.planner.specialist_hints),
        rotation_classes=config.auxiliary_heads.rotation_classes,
        handwriting_classes=config.auxiliary_heads.handwriting_classes,
    )

    outputs = model(batch.image, batch.decoder_input_ids, visual_mode=stage.visual_mode)
    report = compute_pretraining_losses(outputs, batch, stage)

    assert report.total_loss.item() > 0.0
    assert "token_ce" in report.loss_terms
    assert "mtp_ce" in report.loss_terms
    assert "formula_ce" in report.loss_terms


def test_pretraining_optimizer_groups_and_materialization(tmp_path) -> None:
    config = LeopardiS0Config.tiny()
    model = LeopardiS0(config)
    stage = PretrainStageConfig(stage="p2_multimodal_core", visual_mode="standard")
    optimizer = build_optimizer(model, stage)

    assert len(optimizer.param_groups) >= 2
    assert optimizer_group_summary(model, stage)

    payload = materialize_pretraining_stage(
        experiment_id="leo-s0-p2-test",
        stage=stage,
        model_config_path="configs/model/leopardi_s0.yaml",
        root=tmp_path / "runs",
    )
    assert payload["plan"]["stage"] == "p2_multimodal_core"
    assert (
        tmp_path / "runs" / "leo-s0-p2-test" / "artifacts" / "pretraining" / "p2_multimodal_core" / "training-plan.json"
    ).exists()


def test_pretraining_stage_recipes_expose_explicit_bundle_flow() -> None:
    p1 = stage_recipe("p1_text_warmup")
    p2 = stage_recipe("p2_multimodal_core")
    p3 = stage_recipe("p3_hard_curriculum")

    assert p1.data_bundle_ids == ("tokenizer_v1", "p1_text_warmup_v1")
    assert p2.data_bundle_ids == ("p2_exact_core_v1", "p2_structural_aux_v1")
    assert p3.data_bundle_ids == ("p2_exact_core_v1", "p2_structural_aux_v1", "p3_hardcases_v1")
    assert p2.data_mix.exact_pairs > p2.data_mix.synthetic_exact
    assert p3.data_mix.synthetic_exact >= 0.30
