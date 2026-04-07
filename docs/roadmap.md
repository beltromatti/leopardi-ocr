# Leopardi Roadmap

Date locked: 2026-04-07

This roadmap turns the research corpus and blueprint into an execution sequence.

The central strategy is explicit:

- move fast with `Leopardi-S0` at `~100M`
- run many disciplined ablations on `RTX 5090`
- discover the best recipe first
- scale to `Leopardi-S1` at `~500M` only after the recipe is locked

## Phase 0. Protocol Lock

Deliverables:

- canonical output format locked
- benchmark protocol locked
- leakage policy locked
- experiment registry and run naming locked

Must produce:

- reproducible normalization scripts
- benchmark harness specification and runners
- model-card template

## Phase 1. Data Engine v1

Deliverables:

- ingestion for arXiv paired data
- ingestion for PMC OA paired data
- auxiliary dataset loaders for layout, tables, formulas, handwriting, forms, and charts
- dataset lineage and dedup system

Must produce:

- `pretrain_exact`
- `pretrain_synthetic`
- `sft_exact`
- `eval_holdout`

## Phase 2. `Leopardi-S0` Baseline

Deliverables:

- first dense `~100M` parser
- block planner
- writer decoder
- canonicalizer

Must answer:

1. does block planning beat flat decoding?
2. what visual token budget is best?
3. what tokenizer works best for Markdown plus LaTeX?

## Phase 3. Specialist Strengthening

Deliverables:

- formula path
- table path
- handwriting path
- chart path

Must answer:

1. which specialist heads are worth their parameter cost?
2. where does the long tail still collapse?

## Phase 4. Repair and Structured Decoding

Deliverables:

- validator stack
- local repair mode
- first `xgrammar` and `llguidance` experiments

Must answer:

1. decode-time constraints, repair-time constraints, or hybrid?
2. how much latency tax buys how much exactness?

## Phase 5. RLVR and Compression

Deliverables:

- `verl` training path
- objective reward suite
- compression-aware serving variants

Must answer:

1. do verifiable rewards improve exactness without destabilizing the parser?
2. which quantization target is survivable for `Leopardi-S0`?

## Phase 6. Frontier-Grade `Leopardi-S0`

Deliverables:

- best `100M` checkpoint
- public-comparison benchmark report
- error taxonomy
- locked blueprint for scale-up

Success criteria:

- dominates the `<150M` class on the chosen metrics
- credible size-normalized challenge to larger competitors

## Phase 7. `Leopardi-S1` Scale-Up

Deliverables:

- `~500M` scaled model using the same architecture family
- same output contract
- same benchmark protocol

Reason to start this phase:

- only after `Leopardi-S0` has taught us the right recipe

## Phase 8. Final Frontier Push

Deliverables:

- full open benchmark campaign
- open and closed competitor comparison
- serving and quantization package
- research report

Success criteria:

- credible claim to best-in-class accuracy-speed frontier
- strongest public Markdown-plus-LaTeX parser in the target regime

## What Not To Do

### 1. Do not jump early to `500M`

That would slow down discovery.

### 2. Do not keep changing the output format midstream

That destroys comparability.

### 3. Do not optimize only clean PDFs

The hard slices are where leadership is won.

### 4. Do not let runtime and compression wait until the end

That is how fast models become slow products.

## Immediate Next Step

The next engineering step after this blueprint is:

1. implement `arXiv` and `PMC OA` data builders and publish the first canonical bundles
2. implement the first end-to-end pretraining loop against those bundles
3. implement the first evaluation runners for `public_frontier_v1` and `internal_holdout_v1`
4. launch the first real `Leopardi-S0` baseline on `RTX 5090`
