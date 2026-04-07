# Leopardi OCR

Leopardi is a research-first OCR and document parsing project aimed at top-tier accuracy and parsing speed on single-page PDFs, with native Markdown output and LaTeX for mathematics.

## Product Target

- Input: one PDF page, arbitrary size, arbitrary rotation, mixed print + handwriting, tables, charts, titles, paragraphs, and formulas.
- Output: structurally correct Markdown with LaTeX for inline and display math.
- Optimization target: accuracy and latency at the same time, not accuracy alone.

## Working Hypothesis

The most credible path to state-of-the-art is not a monolithic OCR stack. Leopardi is scaffolded around:

1. geometry normalization for arbitrary page rotation and skew
2. high-resolution document parsing with a fast draft pass
3. specialist experts for formulas, tables, charts, and handwriting
4. constrained Markdown + LaTeX decoding
5. repair and verification loops only when confidence drops

That architecture is described in [docs/architecture.md](./docs/architecture.md), benchmarked in [docs/benchmarks.md](./docs/benchmarks.md), and grounded in recent literature in [docs/research/landscape-2026-04.md](./docs/research/landscape-2026-04.md).

## Research Hub

The first competitive intelligence phase is now tracked in:

- [docs/research/README.md](./docs/research/README.md)
- [docs/research/competitor-landscape-2026-04.md](./docs/research/competitor-landscape-2026-04.md)
- [docs/research/leaderboard-2026-04.md](./docs/research/leaderboard-2026-04.md)
- [docs/research/unified-metrics.md](./docs/research/unified-metrics.md)
- [docs/research/model-dossiers.md](./docs/research/model-dossiers.md)
- [docs/research/sources.md](./docs/research/sources.md)

## Repository Layout

- `data_pipeline/`: ingestion, synthesis, curation, and manifests for large-scale pretraining and finetuning data.
- `evaluation/`: metrics, task definitions, and baseline protocol.
- `benchmark/`: benchmark runners and dataset-specific adapters.
- `pretraining/`: curriculum and objectives for large-scale synthetic + paired document pretraining.
- `finetune/`: supervised and preference-based alignment stages.
- `configs/`: shared experiment configuration.
- `src/leopardi/`: Python package, CLI, and output schema.
- `docs/`: architecture, roadmap, benchmarks, and paper references.

## Quickstart

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"
pytest
python -m leopardi.cli doctor
```

## First Milestones

- Build a fast single-page parser baseline with constrained Markdown decoding.
- Add specialist evaluation for math, tables, and reading order.
- Scale synthetic pretraining from PDF-source pairs and rendered perturbations.
- Introduce a latency-aware draft-then-verify decoding path.
