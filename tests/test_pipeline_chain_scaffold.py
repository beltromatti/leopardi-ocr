from __future__ import annotations

import json
from pathlib import Path

from leopardi.data_pipeline import build_data_build_execution_plan
from leopardi.data_pipeline.config import DataBuildStageConfig
from leopardi.evaluation.config import EvaluationStageConfig
from leopardi.evaluation.runtime import materialize_evaluation_stage
from leopardi.finetune.config import FinetuneStageConfig
from leopardi.finetune.runtime import materialize_finetune_stage
from leopardi.inference.config import InferenceStageConfig
from leopardi.inference.runtime import materialize_inference_stage
from leopardi.model import LeopardiS0Config
from leopardi.optimization.config import OptimizationStageConfig
from leopardi.optimization.runtime import materialize_optimization_stage
from leopardi.pretraining.config import PretrainStageConfig
from leopardi.pretraining.runtime import materialize_pretraining_stage


def _read_json(path: Path) -> dict[str, object]:
    return json.loads(path.read_text(encoding="utf-8"))


def test_pipeline_chain_is_coherent(tmp_path: Path) -> None:
    root = tmp_path / "runs"

    pretrain_data_stage = DataBuildStageConfig.from_yaml(
        "configs/data/s0_pretrain_family_build.yaml",
        "configs/runtime/data_build_rtx5090.yaml",
    )
    finetune_foundation_stage = DataBuildStageConfig.from_yaml(
        "configs/data/s0_finetune_foundation_build.yaml",
        "configs/runtime/data_build_rtx5090.yaml",
    )
    finetune_followup_stage = DataBuildStageConfig.from_yaml(
        "configs/data/s0_finetune_followup_build.yaml",
        "configs/runtime/data_build_rtx5090.yaml",
    )
    pretrain_stage = PretrainStageConfig.from_yaml(
        "configs/pretraining/s0_p2_multimodal_core.yaml",
        "configs/runtime/train_rtx5090.yaml",
    )
    finetune_stage = FinetuneStageConfig.from_yaml(
        "configs/finetune/s0_f3_rlvr.yaml",
        "configs/runtime/finetune_rtx5090.yaml",
    )
    optimization_stage = OptimizationStageConfig.from_yaml(
        "configs/optimization/s0_o2_vllm_compressed.yaml",
        "configs/runtime/optimization_rtx5090.yaml",
    )
    inference_stage = InferenceStageConfig.from_yaml(
        "configs/inference/s0_i1_vllm_adaptive.yaml",
        "configs/runtime/inference_rtx5090.yaml",
    )
    evaluation_stage = EvaluationStageConfig.from_yaml(
        "configs/eval/public_frontier.yaml",
        "configs/runtime/eval_rtx5090.yaml",
    )

    pretrain_data_plan = build_data_build_execution_plan(
        experiment_id="data-plan-preview-pretrain",
        stage=pretrain_data_stage,
        stage_config_path="configs/data/s0_pretrain_family_build.yaml",
        runtime_config_path="configs/runtime/data_build_rtx5090.yaml",
        root=root,
    )
    finetune_foundation_plan = build_data_build_execution_plan(
        experiment_id="data-plan-preview-finetune-foundation",
        stage=finetune_foundation_stage,
        stage_config_path="configs/data/s0_finetune_foundation_build.yaml",
        runtime_config_path="configs/runtime/data_build_rtx5090.yaml",
        root=root,
    )
    finetune_followup_plan = build_data_build_execution_plan(
        experiment_id="data-plan-preview-finetune-followup",
        stage=finetune_followup_stage,
        stage_config_path="configs/data/s0_finetune_followup_build.yaml",
        runtime_config_path="configs/runtime/data_build_rtx5090.yaml",
        root=root,
    )
    built_bundles = (
        set(pretrain_data_plan.bundle_ids)
        | set(finetune_foundation_plan.bundle_ids)
        | set(finetune_followup_plan.bundle_ids)
    )

    assert set(pretrain_stage.data_bundle_ids).issubset(built_bundles)
    assert set(finetune_stage.data_bundle_ids).issubset(built_bundles)
    assert optimization_stage.calibration.bundle_id in built_bundles

    pretrain_payload = materialize_pretraining_stage(
        experiment_id="leo-s0-p2-chain-test",
        stage=pretrain_stage,
        model_config_path="configs/model/leopardi_s0.yaml",
        stage_config_path="configs/pretraining/s0_p2_multimodal_core.yaml",
        runtime_config_path="configs/runtime/train_rtx5090.yaml",
        root=root,
    )
    pretrain_manifest = _read_json(root / "leo-s0-p2-chain-test" / "manifest.json")
    assert tuple(pretrain_manifest["data_bundle_ids"]) == pretrain_stage.data_bundle_ids
    assert tuple(pretrain_payload["plan"]["data_bundle_ids"]) == pretrain_stage.data_bundle_ids

    finetune_payload = materialize_finetune_stage(
        experiment_id="leo-s0-f3-chain-test",
        stage=finetune_stage,
        model_config_path="configs/model/leopardi_s0.yaml",
        stage_config_path="configs/finetune/s0_f3_rlvr.yaml",
        runtime_config_path="configs/runtime/finetune_rtx5090.yaml",
        root=root,
    )
    finetune_manifest = _read_json(root / "leo-s0-f3-chain-test" / "manifest.json")
    assert tuple(finetune_manifest["data_bundle_ids"]) == finetune_stage.data_bundle_ids
    assert tuple(finetune_payload["plan"]["data_bundle_ids"]) == finetune_stage.data_bundle_ids

    finetune_checkpoint_uri = str(finetune_payload["plan"]["checkpoint_uri"])
    optimization_payload = materialize_optimization_stage(
        experiment_id="leo-s0-o2-chain-test",
        stage=optimization_stage,
        base_checkpoint_uri=finetune_checkpoint_uri,
        stage_config_path="configs/optimization/s0_o2_vllm_compressed.yaml",
        runtime_config_path="configs/runtime/optimization_rtx5090.yaml",
        root=root,
    )
    optimization_manifest = _read_json(root / "leo-s0-o2-chain-test" / "manifest.json")
    assert optimization_manifest["data_bundle_ids"] == [optimization_stage.calibration.bundle_id]

    optimization_variants = {plan["variant_id"]: plan for plan in optimization_payload["plans"]}
    assert inference_stage.artifact_variant_id in optimization_variants
    assert inference_stage.runtime_family in optimization_variants[inference_stage.artifact_variant_id]["runtime_targets"]

    inference_payload = materialize_inference_stage(
        experiment_id="leo-s0-i1-chain-test",
        stage=inference_stage,
        stage_config_path="configs/inference/s0_i1_vllm_adaptive.yaml",
        runtime_config_path="configs/runtime/inference_rtx5090.yaml",
        root=root,
    )
    inference_runtime_families = {plan["runtime_family"] for plan in inference_payload["plans"]}
    assert evaluation_stage.runtime.primary_runtime in inference_runtime_families
    assert evaluation_stage.runtime.alternate_runtime in inference_runtime_families
    assert set(evaluation_stage.decode_modes).issubset({mode.name for mode in inference_stage.modes})

    evaluation_payload = materialize_evaluation_stage(
        experiment_id="leo-s0-eval-chain-test",
        stage=evaluation_stage,
        stage_config_path="configs/eval/public_frontier.yaml",
        runtime_config_path="configs/runtime/eval_rtx5090.yaml",
        root=root,
    )
    evaluation_manifest = _read_json(root / "leo-s0-eval-chain-test" / "manifest.json")
    assert evaluation_manifest["data_bundle_ids"] == [evaluation_stage.bundle_id]
    assert evaluation_payload["dataset_count"] > 0
    assert evaluation_payload["baseline_count"] > 0

    model_config = LeopardiS0Config.from_yaml("configs/model/leopardi_s0.yaml")
    assert model_config.target_params_m <= 200
