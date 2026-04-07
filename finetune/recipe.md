# Finetune Recipe

Date locked: 2026-04-08

This is the first serious finetuning recipe for `Leopardi-S0`.

## Stage Order

1. `F0` high-quality general SFT
2. `F1` specialist SFT on hard slices
3. `F2` local repair SFT
4. `F3` RLVR with objective rewards

## Why This Order

`F0` creates stable canonical behavior.
`F1` moves the long tail.
`F2` reduces expensive full re-decodes.
`F3` is last because RL only helps if the model already writes coherent canonical outputs.

## Default Model-Update Policy

### `F0`

- full finetune

Why:

- `S0` is small enough that full tuning is feasible on one `RTX 5090`
- we want to move the whole model into the exact-output regime

### `F1`

- full finetune

Why:

- specialist slices should reshape shared representations, not only add a thin adapter

### `F2`

- full finetune

Why:

- repair quality depends on tight integration between planner, decoder, and auxiliary heads

### `F3`

- LoRA-style RL update path by default

Why:

- RL on one rented GPU benefits from lower memory pressure and faster branching
- current open RL stacks such as `verl` explicitly support LoRA-style post-training

## Release Discipline

- do not skip `F2` if repair is part of the deployment design
- do not start `F3` before SFT is stable
- do not mix benchmark-derived pages into finetune bundles
