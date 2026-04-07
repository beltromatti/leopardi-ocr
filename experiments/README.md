# Experiments

This directory is the operational backbone for Leopardi's multi-experiment workflow.

It exists to prevent the project from collapsing into:

- untraceable runs
- inconsistent naming
- benchmark leakage
- accidental redefinition of the "best" model

## Lifecycle

Every experiment belongs to exactly one lifecycle state:

1. `draft`
   - proposed but not started
2. `active`
   - currently running or being evaluated
3. `candidate`
   - finished and worth benchmark comparison
4. `promoted`
   - best current representative for a track
5. `frozen`
   - locked reference for reproducibility
6. `archived`
   - kept for history but not part of active decision-making

## Main Subdirectories

- `registry/`: master experiment index and promotion history
- `templates/`: templates for experiment specs, ablations, reports, and release cards
- `tracks/`: track definitions for `Leopardi-S0`, `Leopardi-S1`, and specialist branches
- `promotions/`: rules and checklists for moving checkpoints between states

## Naming Convention

Recommended experiment id:

`leo-<track>-<stage>-<family>-<yyyymmdd>-<serial>`

Examples:

- `leo-s0-p2-dense-20260408-001`
- `leo-s0-f1-table-20260412-003`
- `leo-s1-f3-rlvr-20260701-002`

Rules:

- `track`: `s0`, `s1`, `table`, `math`, `repair`, `runtime`
- `stage`: `p0`, `p1`, `p2`, `p3`, `f0`, `f1`, `f2`, `f3`, `eval`, `serve`
- `family`: short architecture or recipe name
- `serial`: 3-digit sequence within the day

## Required Metadata

Every experiment must declare:

- experiment id
- parent experiment ids
- objective
- config stack
- data sources and split version
- benchmark protocol version
- hardware tag
- checkpoint lineage
- run manifest location
- persistence targets
- decision outcome

## Golden Rule

No experiment result is considered real unless it can be located from:

- the registry
- the config stack
- the benchmark protocol version
- the dataset split version
- the run manifest and artifact ledger
