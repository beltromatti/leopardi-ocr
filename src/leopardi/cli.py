from __future__ import annotations

from pathlib import Path

import typer
from rich.console import Console

from leopardi.model import LeopardiS0
from leopardi.pretraining.batch import PretrainBatch
from leopardi.pretraining.config import PretrainStageConfig
from leopardi.pretraining.losses import compute_pretraining_losses
from leopardi.schemas.output import ParsedPage

app = typer.Typer(help="Leopardi OCR developer CLI.")
console = Console()


@app.command()
def doctor() -> None:
    console.print("[bold green]Leopardi scaffold is healthy.[/bold green]")
    console.print(f"Repository root: {Path.cwd()}")


@app.command()
def schema_example() -> None:
    page = ParsedPage.example()
    console.print(page.model_dump_json(indent=2))


@app.command()
def benchmark(checkpoint: str = typer.Argument("draft")) -> None:
    console.print(f"Benchmark runner scaffold for checkpoint: [bold]{checkpoint}[/bold]")
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


if __name__ == "__main__":
    app()
