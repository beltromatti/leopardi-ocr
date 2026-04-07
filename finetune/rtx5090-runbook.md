# RTX 5090 Finetune Runbook

Date locked: 2026-04-08

This file defines the practical finetuning policy for a rented single `RTX 5090`.

## Hardware Reality

- one GPU
- finite and ephemeral local storage
- interruptions are possible

## SFT Policy

For `F0` to `F2`:

- `bf16`
- gradient checkpointing on
- micro-batch conservative
- accumulate gradients to reach useful effective batch sizes
- save and validate often

## RLVR Policy

For `F3`:

- default to LoRA-style updates
- keep rollouts short and targeted
- use a small but well-curated prompt pack
- prefer structural rewards to noisy heuristic rewards

## Data Policy

- use already-published finetune bundles whenever possible
- do not rebuild all data on the finetune machine unless unavoidable
- stage local copies and purge after checkpoint publication and verification

## Runtime Policy

- `vLLM` is the safest rollout baseline
- `SGLang` is the high-performance path once the reward loop is stable
- keep one runtime primary per experiment so comparisons remain honest
