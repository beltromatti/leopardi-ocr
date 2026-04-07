# Leopardi OCR

Leopardi is a research-first OCR and document parsing project aimed at top-tier accuracy and parsing speed on full documents, with native Markdown output and LaTeX for mathematics.

## Product Target

- Input: full documents and document images with arbitrary size, arbitrary rotation, mixed print + handwriting, tables, charts, titles, paragraphs, and formulas.
- Output: structurally correct Markdown with LaTeX for inline and display math.
- Optimization target: accuracy and latency at the same time, not accuracy alone.

In practice, many frontier systems still process documents through page-level or region-level units internally. Leopardi follows that reality: document parsing is the product target, while page and region parsing remain core internal abstractions for training, routing, and evaluation.

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
- [docs/research/open-source-codebase-audit-2026-04.md](./docs/research/open-source-codebase-audit-2026-04.md)
- [docs/research/leopardi-directions-2026-04.md](./docs/research/leopardi-directions-2026-04.md)
- [docs/research/sources.md](./docs/research/sources.md)

Open-source competitor repos are vendored as submodules under [external/competitors/README.md](./external/competitors/README.md) for codebase-driven analysis and future reproducible comparisons.

## Repository Layout

- `experiments/`: experiment registry, templates, track definitions, and promotion rules.
- `data_pipeline/`: ingestion, synthesis, curation, and manifests for large-scale pretraining and finetuning data.
- `evaluation/`: unified evaluation system including datasets, protocols, runners, metrics, baselines, and reports.
- `pretraining/`: curriculum and objectives for large-scale synthetic + paired document pretraining.
- `finetune/`: supervised and preference-based alignment stages.
- `ops/`: shared run, logging, recovery, and persistence contract for all phases.
- `configs/`: shared experiment configuration.
- `src/leopardi/`: Python package, CLI, and output schema.
- `docs/`: architecture, roadmap, benchmarks, and paper references.

## Experiment System

Leopardi is set up for many disciplined experiments rather than a few ad hoc runs.

Start here:

- [docs/experimentation.md](./docs/experimentation.md)
- [docs/run-on-rtx5090.md](./docs/run-on-rtx5090.md)
- [experiments/README.md](./experiments/README.md)
- [configs/README.md](./configs/README.md)

The intended workflow is:

1. define a layered config stack
2. register the experiment
3. run and evaluate it under a pinned evaluation protocol
4. promote only after explicit checklist review

## Quickstart

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e ".[dev,train]"
pytest -q
ruff check src tests ops docs configs experiments evaluation data_pipeline pretraining finetune
python -m leopardi.cli doctor
```

## Current Readiness

Leopardi is currently ready as a research and run-control workspace.

That means:

- the model, config, runtime, logging, and persistence scaffolds are in place
- the experiment system and benchmark protocol surface are defined
- the rented-machine workflow is documented

What is still missing before the first full training campaign:

- real data builders
- real end-to-end train and finetune loops
- real evaluation runners

## First Milestones

- Build a fast document parser baseline with constrained Markdown decoding.
- Add specialist evaluation for math, tables, and reading order.
- Scale synthetic pretraining from PDF-source pairs and rendered perturbations.
- Introduce a latency-aware draft-then-verify decoding path.
