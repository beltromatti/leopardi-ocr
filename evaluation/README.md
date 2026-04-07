# Evaluation

Date locked: 2026-04-08

This directory is the complete scientific measurement stack for Leopardi.

It is responsible for one thing:

- turning model outputs into defensible claims

It is not a training directory, a data-building directory, or a miscellaneous utility directory.
It is the place where Leopardi defines:

- what is being measured
- how it is being measured
- what counts as a fair comparison
- when a result is strong enough to be promoted

## Evaluation Principles

### 1. One home for all measurement logic

`evaluation/` contains the full measurement surface:

- dataset families
- protocol versions
- task schemas
- metric definitions
- competitor baselines
- immutable reports

### 2. Protocols are versioned and stable

A promoted result is always tied to:

- dataset bundle
- protocol version
- hardware tag
- decode mode
- canonical output contract

### 3. Public and internal evidence are both required

Leopardi should never optimize only for public leaderboard appearance.

Every serious result must include:

- public benchmark evidence
- internal holdout evidence
- failure-slice review
- latency evidence

### 4. Benchmark families are not interchangeable

The system explicitly distinguishes:

- document parsing
- PDF-to-Markdown extraction
- formula exactness
- table structure
- handwriting and rotation
- multilingual and photographed robustness

### 5. Output normalization is first-class

Leopardi outputs Markdown plus LaTeX.
That means evaluation must normalize format variation without forgiving structural failure.

## Directory Map

- `datasets/`
  - dataset registry, bundles, split roles, and benchmark-family notes
- `protocols/`
  - pinned protocol versions used for public claims, internal review, competitor reproduction, and release gates
- `tasks/`
  - task model and task bundles that map benchmark families to expected outputs
- `metrics/`
  - metric catalog, normalization rules, and scorecards
- `baselines/`
  - competitor registry and reproduction policy
- `runners/`
  - execution contract and artifact expectations for future implementations
- `reports/`
  - immutable report contract and templates

## Shared Operations

`evaluation/` follows the common run and persistence contract defined in:

- `ops/`

## Relation To The Rest Of The Repo

- `docs/benchmarks.md` defines the high-level benchmark blueprint
- `docs/research/unified-metrics.md` defines the research-layer metric synthesis
- `docs/research/leaderboard-2026-04.md` defines the current competitor frontier
- `data_pipeline/` defines training data and split provenance
- `experiments/` defines run identity and promotion history

`evaluation/` is the operational bridge between all of them.

## Minimum Rule For Any Serious Result

No result counts as a Leopardi result unless all are known:

1. experiment id
2. protocol version
3. dataset bundle
4. hardware tag
5. decode mode
6. output normalization rules
7. report location
