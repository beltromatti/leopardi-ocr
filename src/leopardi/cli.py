from __future__ import annotations

from dataclasses import asdict
from pathlib import Path

import typer
from rich.console import Console

from leopardi.finetune.batch import FinetuneBatch
from leopardi.finetune.config import FinetuneStageConfig
from leopardi.finetune.losses import compute_finetune_losses
from leopardi.finetune.rewards import compute_reward_breakdown
from leopardi.model import LeopardiS0
from leopardi.optimization import (
    OptimizationGoalConfig,
    OptimizationStageConfig,
    VariantMeasurement,
    build_variant_runtime_plan,
    build_variant_summary,
    materialize_optimization_stage,
    optimization_stage_recipe_dict,
    pareto_frontier,
    rank_candidates,
)
from leopardi.ops import (
    RunHeartbeat,
    RunManifest,
    RunSummary,
    append_event,
    build_run_layout,
    ensure_run_layout,
    write_heartbeat,
    write_manifest,
    write_summary,
)
from leopardi.pretraining.batch import PretrainBatch
from leopardi.pretraining.config import PretrainStageConfig
from leopardi.pretraining.losses import compute_pretraining_losses
from leopardi.schemas.output import ParsedPage

app = typer.Typer(help="Leopardi OCR developer CLI.")
console = Console()


@app.command()
def doctor() -> None:
    console.print("[bold green]Leopardi control plane is healthy.[/bold green]")
    console.print(f"Repository root: {Path.cwd()}")


@app.command()
def schema_example() -> None:
    page = ParsedPage.example()
    console.print(page.model_dump_json(indent=2))


@app.command()
def benchmark(checkpoint: str = typer.Argument("draft")) -> None:
    console.print(f"Evaluation runners are not implemented yet for checkpoint: [bold]{checkpoint}[/bold]")
    console.print(
        "Implement dataset adapters in evaluation/datasets and runners in evaluation/runners."
    )


@app.command()
def model_summary(
    config_path: Path = typer.Argument(Path("configs/model/leopardi_s0.yaml")),
) -> None:
    model = LeopardiS0.from_yaml(str(config_path))
    console.print(model.summary())


@app.command()
def pretrain_summary(
    stage_config: Path = typer.Argument(Path("configs/pretraining/s0_p2_multimodal_core.yaml")),
    runtime_config: Path = typer.Argument(Path("configs/runtime/train_rtx5090.yaml")),
) -> None:
    stage = PretrainStageConfig.from_yaml(stage_config, runtime_config)
    console.print(stage)


@app.command()
def smoke_train_step(
    model_config: Path = typer.Argument(Path("configs/model/leopardi_s0.yaml")),
    stage_config: Path = typer.Argument(Path("configs/pretraining/s0_p2_multimodal_core.yaml")),
    runtime_config: Path = typer.Argument(Path("configs/runtime/train_rtx5090.yaml")),
) -> None:
    model = LeopardiS0.from_yaml(str(model_config))
    stage = PretrainStageConfig.from_yaml(stage_config, runtime_config)
    batch = PretrainBatch.synthetic(
        batch_size=1,
        image_size=(256, 256),
        seq_len=64,
        vocab_size=model.config.writer_decoder.vocab_size,
        planner_blocks=model.config.planner.num_blocks,
        visual_tokens=sum(
            grid[0] * grid[1] for grid in model.config.visual_tokenizer.pool_layouts[stage.visual_mode]
        ),
        num_block_types=len(model.config.planner.block_types),
        num_length_buckets=model.config.planner.num_length_buckets,
        num_hints=len(model.config.planner.specialist_hints),
        rotation_classes=model.config.auxiliary_heads.rotation_classes,
        handwriting_classes=model.config.auxiliary_heads.handwriting_classes,
    )
    outputs = model(batch.image, batch.decoder_input_ids, visual_mode=stage.visual_mode)
    report = compute_pretraining_losses(outputs, batch, stage)
    console.print(
        {
            "model": model.summary(),
            "loss_report": report.loss_terms,
            "total": float(report.total_loss.detach()),
        }
    )


@app.command()
def finetune_summary(
    stage_config: Path = typer.Argument(Path("configs/finetune/s0_f0_sft.yaml")),
    runtime_config: Path = typer.Argument(Path("configs/runtime/finetune_rtx5090.yaml")),
) -> None:
    stage = FinetuneStageConfig.from_yaml(stage_config, runtime_config)
    console.print(stage)


@app.command()
def smoke_finetune_step(
    model_config: Path = typer.Argument(Path("configs/model/leopardi_s0.yaml")),
    stage_config: Path = typer.Argument(Path("configs/finetune/s0_f0_sft.yaml")),
    runtime_config: Path = typer.Argument(Path("configs/runtime/finetune_rtx5090.yaml")),
) -> None:
    model = LeopardiS0.from_yaml(str(model_config))
    stage = FinetuneStageConfig.from_yaml(stage_config, runtime_config)
    visual_tokens = sum(
        grid[0] * grid[1] for grid in model.config.visual_tokenizer.pool_layouts[stage.visual_mode]
    )
    batch = FinetuneBatch.synthetic(
        batch_size=1,
        image_size=(256, 256),
        seq_len=64,
        vocab_size=model.config.writer_decoder.vocab_size,
        planner_blocks=model.config.planner.num_blocks,
        visual_tokens=visual_tokens,
        num_block_types=len(model.config.planner.block_types),
        num_length_buckets=model.config.planner.num_length_buckets,
        num_hints=len(model.config.planner.specialist_hints),
        rotation_classes=model.config.auxiliary_heads.rotation_classes,
        handwriting_classes=model.config.auxiliary_heads.handwriting_classes,
    )
    outputs = model(batch.image, batch.decoder_input_ids, visual_mode=stage.visual_mode)
    loss_report = compute_finetune_losses(outputs, batch, stage)
    reward_report = compute_reward_breakdown(batch.reward_signals or {}, stage)
    console.print(
        {
            "model": model.summary(),
            "loss_report": loss_report.loss_terms,
            "total_loss": float(loss_report.total_loss.detach()),
            "reward_report": reward_report.reward_terms,
            "total_reward": float(reward_report.total_reward.detach()),
        }
    )


@app.command()
def run_layout(experiment_id: str = typer.Argument("leo-s0-p2-dense-20260408-001")) -> None:
    console.print(build_run_layout(experiment_id).as_dict())


@app.command()
def ops_examples() -> None:
    manifest = RunManifest(
        experiment_id="leo-s0-p2-dense-20260408-001",
        phase="pretraining",
        stage="p2_multimodal_core",
        track="s0-core",
        hardware_tag="rtx5090",
        config_paths=[
            "configs/model/leopardi_s0.yaml",
            "configs/pretraining/s0_p2_multimodal_core.yaml",
            "configs/runtime/train_rtx5090.yaml",
        ],
        data_bundle_ids=["p2_exact_core_v1", "p2_structural_aux_v1"],
        protocol_version="internal_holdout_v1",
        local_run_root="runs/leo-s0-p2-dense-20260408-001",
        persistent_targets={
            "checkpoints": "hf://leopardi-ocr-checkpoints",
            "reports": "hf://leopardi-ocr-reports",
        },
    )
    heartbeat = RunHeartbeat(
        experiment_id=manifest.experiment_id,
        phase=manifest.phase,
        stage=manifest.stage,
        state="running",
        current_step=1280,
        latest_metrics={"loss": 1.23, "eval_markdown_validity": 0.91},
        last_save_step=1000,
        last_save_at="2026-04-08T12:34:56Z",
        last_sync_at="2026-04-08T12:35:10Z",
        last_sync_status="ok",
    )
    summary = RunSummary(
        experiment_id=manifest.experiment_id,
        phase=manifest.phase,
        stage=manifest.stage,
        outcome="completed",
        key_metrics={"loss": 0.98, "eval_markdown_validity": 0.93},
    )
    console.print(
        {
            "manifest": manifest.model_dump(),
            "heartbeat": heartbeat.model_dump(),
            "summary": summary.model_dump(),
        }
    )


@app.command("materialize-run-example")
def materialize_run_example(
    experiment_id: str = typer.Argument("leo-s0-p2-dense-20260408-001"),
    root: Path = typer.Option(Path("runs"), "--root"),
) -> None:
    layout = ensure_run_layout(experiment_id, root=root)
    manifest = RunManifest(
        experiment_id=experiment_id,
        phase="pretraining",
        stage="p2_multimodal_core",
        track="s0-core",
        hardware_tag="rtx5090",
        config_paths=[
            "configs/model/leopardi_s0.yaml",
            "configs/pretraining/s0_p2_multimodal_core.yaml",
            "configs/runtime/train_rtx5090.yaml",
        ],
        data_bundle_ids=["p2_exact_core_v1", "p2_structural_aux_v1"],
        protocol_version="internal_holdout_v1",
        local_run_root=str(layout.experiment_root),
        persistent_targets={
            "checkpoints": "hf://leopardi-ocr-checkpoints",
            "reports": "hf://leopardi-ocr-reports",
        },
    )
    heartbeat = RunHeartbeat(
        experiment_id=experiment_id,
        phase="pretraining",
        stage="p2_multimodal_core",
        state="running",
        current_step=1280,
        latest_metrics={"loss": 1.23, "eval_markdown_validity": 0.91},
        last_save_step=1000,
        last_save_at="2026-04-08T12:34:56Z",
        last_sync_at="2026-04-08T12:35:10Z",
        last_sync_status="ok",
    )
    summary = RunSummary(
        experiment_id=experiment_id,
        phase="pretraining",
        stage="p2_multimodal_core",
        outcome="completed",
        key_metrics={"loss": 0.98, "eval_markdown_validity": 0.93},
    )
    write_manifest(manifest, layout=layout)
    write_heartbeat(heartbeat, layout=layout)
    write_summary(summary, layout=layout)
    append_event(
        layout=layout,
        event_type="run_initialized",
        phase=manifest.phase,
        stage=manifest.stage,
        payload={"track": manifest.track, "hardware_tag": manifest.hardware_tag},
    )
    console.print(layout.as_dict())


@app.command()
def optimization_summary(
    stage_config: Path = typer.Argument(Path("configs/optimization/s0_o2_vllm_compressed.yaml")),
    runtime_config: Path = typer.Argument(Path("configs/runtime/optimization_rtx5090.yaml")),
) -> None:
    stage = OptimizationStageConfig.from_yaml(stage_config, runtime_config)
    console.print(build_variant_summary(stage))


@app.command()
def optimization_recipes() -> None:
    console.print(
        {
            recipe: optimization_stage_recipe_dict(recipe)
            for recipe in (
                "o0_reference_export",
                "o1_torchao_portable",
                "o2_vllm_compressed",
                "o3_runtime_kv",
                "o4_qat_export",
            )
        }
    )


@app.command()
def optimization_rank_example() -> None:
    goal = OptimizationGoalConfig()
    reference = VariantMeasurement(
        variant_id="bf16_reference",
        overall_score=0.945,
        markdown_validity=0.998,
        latex_validity=0.994,
        table_score=0.928,
        formula_score=0.941,
        latency_ms=1200.0,
        peak_memory_gb=22.0,
        throughput_pages_per_second=0.83,
    )
    candidates = [
        VariantMeasurement(
            variant_id="llmcompressor_fp8_dynamic",
            overall_score=0.941,
            markdown_validity=0.997,
            latex_validity=0.992,
            table_score=0.925,
            formula_score=0.938,
            latency_ms=930.0,
            peak_memory_gb=16.5,
            throughput_pages_per_second=1.08,
        ),
        VariantMeasurement(
            variant_id="torchao_int4_weight_only",
            overall_score=0.934,
            markdown_validity=0.996,
            latex_validity=0.99,
            table_score=0.919,
            formula_score=0.931,
            latency_ms=760.0,
            peak_memory_gb=12.0,
            throughput_pages_per_second=1.31,
        ),
        VariantMeasurement(
            variant_id="vllm_fp8_kv",
            overall_score=0.943,
            markdown_validity=0.997,
            latex_validity=0.993,
            table_score=0.927,
            formula_score=0.939,
            latency_ms=1080.0,
            peak_memory_gb=18.0,
            throughput_pages_per_second=0.93,
        ),
    ]
    console.print(
        {
            "ranked": [asdict(item) for item in rank_candidates(reference, candidates, goal)],
            "pareto_frontier": [asdict(item) for item in pareto_frontier([reference, *candidates])],
        }
    )


@app.command()
def optimization_plan(
    stage_config: Path = typer.Argument(Path("configs/optimization/s0_o2_vllm_compressed.yaml")),
    runtime_config: Path = typer.Argument(Path("configs/runtime/optimization_rtx5090.yaml")),
    base_checkpoint_uri: str = typer.Argument("hf://leopardi-ocr-checkpoints/leo-s0-f3-candidate"),
) -> None:
    stage = OptimizationStageConfig.from_yaml(stage_config, runtime_config)
    plans = []
    for variant in stage.variants:
        plan, _ = build_variant_runtime_plan(
            experiment_id="optimization-plan-preview",
            stage=stage,
            variant=variant,
            artifacts_root="runs/optimization-plan-preview/artifacts",
            persistent_root="hf://leopardi-ocr-checkpoints",
            base_checkpoint_uri=base_checkpoint_uri,
        )
        plans.append(asdict(plan))
    console.print(
        {
            "stage": stage.stage,
            "base_checkpoint_uri": base_checkpoint_uri,
            "plans": plans,
        }
    )


@app.command()
def optimization_materialize(
    experiment_id: str = typer.Argument("leo-s0-o2-opt-20260408-001"),
    stage_config: Path = typer.Argument(Path("configs/optimization/s0_o2_vllm_compressed.yaml")),
    runtime_config: Path = typer.Argument(Path("configs/runtime/optimization_rtx5090.yaml")),
    base_checkpoint_uri: str = typer.Argument("hf://leopardi-ocr-checkpoints/leo-s0-f3-candidate"),
    root: Path = typer.Option(Path("runs"), "--root"),
) -> None:
    stage = OptimizationStageConfig.from_yaml(stage_config, runtime_config)
    console.print(
        materialize_optimization_stage(
            experiment_id=experiment_id,
            stage=stage,
            base_checkpoint_uri=base_checkpoint_uri,
            stage_config_path=str(stage_config),
            runtime_config_path=str(runtime_config),
            root=root,
        )
    )


if __name__ == "__main__":
    app()
