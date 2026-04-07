# Adapter Policy

Date locked: 2026-04-08

Leopardi-S0 is small enough that full finetuning is feasible and often preferable.

This file defines when not to use it.

## Default Policy

- `F0` to `F2`: full finetune
- `F3`: LoRA-style RL updates by default

## Why Full Finetune First

- `S0` is only `~93M`
- full tuning is realistic on one `RTX 5090`
- exact output behavior often needs distributed changes across the whole model

## Why LoRA For RL By Default

- RL iteration is expensive and unstable
- adapter updates reduce memory pressure
- current open RL stacks explicitly support LoRA pathways
- LoRA RL makes branching experiments cheaper

## When To Use LoRA Earlier

Allowed when:

- running many ablations in parallel
- memory becomes the bottleneck
- testing narrow specialist hypotheses

Not ideal when:

- the base model still fails broadly on canonical output

## Compression Interaction

LoRA and QAT are not substitutes for each other.

Use:

- full or LoRA finetune to reach quality
- QAT later to preserve quality under low-bit serving targets
