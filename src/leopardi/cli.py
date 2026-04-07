from __future__ import annotations

from pathlib import Path

import typer
from rich.console import Console

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


if __name__ == "__main__":
    app()
