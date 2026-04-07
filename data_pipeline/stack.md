# Data Platform Stack

Date locked: 2026-04-08

This file defines the preferred practical stack for Leopardi data builds.

## Primary Stack

### Persistent artifact store

- Hugging Face dataset repositories

Use for:

- published training shards
- Parquet manifests
- data cards
- stats bundles

Why:

- versioned publication flow
- easy downstream consumption
- practical for ephemeral build machines

### Training payload format

- WebDataset-style tar shards

Use for:

- image-plus-text training samples
- bundled page payloads without tiny-file explosion

Why:

- common in multimodal training
- consistent with frontier training practice
- easy to mirror into larger future training stacks

### Manifest and analysis format

- Parquet as the bulk metadata layer
- CSV and Markdown as the git-visible summary layer

Why:

- Parquet is compact and analysis-friendly
- CSV and Markdown are review-friendly

### Optional scaling path

- object-store mirror
- Megatron-Energon-style shard ingestion later if scale justifies it

For `Leopardi-S0`, this remains optional.
The control-plane design should not depend on it.

## Local Build Stack

Default local categories:

- metadata cache
- raw cache
- work cache
- upload staging

The local stack must support:

- metadata-first acquisition
- partial rebuilds
- persist-then-purge

## Why This Stack Fits Leopardi

It matches the actual problem:

- small-model rapid iteration
- multimodal sample payloads
- rented hardware with bounded storage
- need for future scale-up without rewriting the data contract
