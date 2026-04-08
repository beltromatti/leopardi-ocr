# RTX 5090 Optimization Runbook

Date locked: 2026-04-08

This file defines the practical optimization policy for a rented single `RTX 5090`.

## Hardware Reality

- one GPU
- finite and ephemeral local storage
- repeated short runs are better than one fragile all-in compression campaign

## First Rule

Always export and validate the `bf16` reference first.

Without it:

- latency deltas are not trustworthy
- quality-retention claims are not trustworthy

## PTQ Policy

Start with:

- `TorchAO` portable variants
- `LLM Compressor` `FP8` and `W4A16` variants for `vLLM`

Do not start with the most aggressive candidate first.

## Calibration Policy

- bounded page count
- hard-slice coverage mandatory
- no evaluation leakage

## Runtime Validation Policy

Every optimized artifact must be rechecked for:

- Markdown validity
- LaTeX validity
- table integrity
- runtime compatibility
- structured-output mode compatibility

## Publication Policy

Publish each deployable variant separately.

Do not overwrite a reference export with a compressed export.
