# Run Contract

Date locked: 2026-04-08

Every serious Leopardi run must follow one directory contract, regardless of phase.

## Run Identity

Every run is keyed by:

- experiment id
- phase
- stage
- hardware tag
- config digest

## Canonical Local Layout

Under local run root `runs/`:

- `<experiment_id>/`
  - `manifest.json`
  - `heartbeat.json`
  - `summary.json`
  - `console.log`
  - `events.ndjson`
  - `control/`
    - `STOP`
    - `RELOAD`
    - `NOTE.txt`
  - `artifacts/`
    - checkpoints
    - reports
    - sample summaries
  - `scratch/`
    - transient local-only work products

## Required Metadata Files

### `manifest.json`

Must include:

- experiment id
- phase
- stage
- config paths
- data bundle identities
- protocol version when applicable
- hardware tag
- local run path
- persistent target locations

### `heartbeat.json`

Must include:

- current state
- wall-clock timestamps
- current step or item count
- last successful save
- latest key metrics
- last sync status

### `summary.json`

Must include:

- final outcome
- artifact pointers
- validation summary
- failure summary

## Phase Mapping

### Data pipeline

- unit: build wave or bundle build

### Pretraining and finetuning

- unit: stage run

### Evaluation

- unit: protocol run

The layout stays the same across all three.
