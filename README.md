# Leopardi OCR

Leopardi is a research-first OCR and document parsing project aimed at top-tier accuracy and parsing speed on full documents, with native Markdown output and LaTeX for mathematics.

## Product Target

- Input: full documents and document images with arbitrary size, arbitrary rotation, mixed print + handwriting, tables, charts, titles, paragraphs, and formulas.
- Output: structurally correct Markdown with LaTeX for inline and display math.
- Optimization target: accuracy and latency at the same time, not accuracy alone.

In practice, many frontier systems still process documents through page-level or region-level units internally. Leopardi follows that reality: document parsing is the product target, while page and region parsing remain core internal abstractions for training, routing, and evaluation.

## Working Hypothesis

The most credible path to state-of-the-art is not a monolithic OCR stack. Leopardi is organized around:

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
- `model/`: model-family control plane, preset policy, interfaces, and artifact rules.
- `data_pipeline/`: ingestion, synthesis, curation, manifests, and operational policy for large-scale pretraining and finetuning data.
- `evaluation/`: unified evaluation system including datasets, protocols, runners, metrics, baselines, and reports.
- `pretraining/`: curriculum and objectives for large-scale synthetic + paired document pretraining.
- `finetune/`: supervised and preference-based alignment stages.
- `optimization/`: post-finetune compression, export, and deployable variant selection.
- `inference/`: runtime, routing, validation, and serving plans for promoted artifacts.
- `ops/`: shared run, logging, recovery, and persistence contract for all phases.
- `configs/`: shared experiment configuration.
- `src/leopardi/`: Python package, CLI, runtime materializers, and output schema.
- `docs/`: architecture, roadmap, benchmarks, and paper references.

## Experiment System

Leopardi is set up for many disciplined experiments rather than a few ad hoc runs.

Start here:

- [docs/experimentation.md](./docs/experimentation.md)
- [docs/run-on-rtx5090.md](./docs/run-on-rtx5090.md)
- [model/README.md](./model/README.md)
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
ruff check src tests ops docs configs experiments evaluation data_pipeline inference pretraining finetune optimization scripts
python -m leopardi.cli doctor
```

For the rented `RTX 5090` machine, use:

- `scripts/bootstrap_rtx5090.sh`
- `scripts/smoke_cpu.sh`
- `scripts/smoke_chain_cpu.sh`

## Current Readiness

Leopardi is currently ready as a research and run-control workspace.

That means:

- the model, config, runtime, logging, persistence, and evaluation scoring layers are in place
- the experiment system and benchmark protocol surface are defined
- the rented-machine workflow is documented

What is still missing before the first full training campaign:

- real end-to-end train and finetune loops
- real optimization export backends
- real inference supervisor that boots the chosen runtime automatically
- real benchmark-specific dataset adapters and automated evaluation supervisors

The data pipeline now includes executable workers for:

- arXiv exact-pair acquisition
- PMC OA exact-pair acquisition
- HF parquet-backed sources such as PubLayNet mirror, MathWriting, Im2LaTeX-100K, FUNSD, CORD, SROIE, FinTabNet-family parquet, ChartQA, and PlotQA
- DocLayNet direct ZIP ingestion
- PubTables-1M structure archives
- SciTSR archive ingestion from the pinned public release

For the current `Leopardi-S0` full external data build on a rented machine, plan for about `400 GB` free disk.
The optimized builder now streams parquet-backed HF sources, processes each source once, and drops local raw plus verified bundle copies as soon as they are no longer needed.

Manual or conditional sources still use a strict local-manifest import contract only for derived internal bundles when they are promoted from prior runs.

Helpful local scripts:

- `scripts/bootstrap_env.sh`
- `scripts/bootstrap_rtx5090.sh`
- `scripts/smoke_cpu.sh`
- `scripts/smoke_chain_cpu.sh`

## First Milestones

- Build a fast document parser baseline with constrained Markdown decoding.
- Add specialist evaluation for math, tables, and reading order.
- Scale synthetic pretraining from PDF-source pairs and rendered perturbations.
- Introduce a latency-aware draft-then-verify decoding path.
