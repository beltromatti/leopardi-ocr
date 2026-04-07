# Artifact Layout

Date locked: 2026-04-08

This file defines how persistent data artifacts should be laid out.

## Recommended Persistent Layout

Use separate dataset repositories or equivalent namespaces for:

- `leopardi-ocr-data-metadata`
- `leopardi-ocr-data-exact`
- `leopardi-ocr-data-synthetic`
- `leopardi-ocr-data-aux`
- `leopardi-ocr-data-bundles`
- `leopardi-ocr-data-holdouts`

These names are conventions, not hard-coded product names.

## Within Each Persistent Dataset Repo

Recommended top-level layout:

- `cards/`
- `manifests/`
- `shards/`
- `stats/`
- `exclusions/`

## Shard Policy

Training payloads should be stored as training-friendly shard files, not as millions of loose images.

Preferred format:

- WebDataset-style tar shards for image-plus-text training samples

Preferred shard sizing:

- medium-sized shards large enough to avoid tiny-file explosion
- small enough to support resume, retry, and partial rebuilds

## Sample Payload Contents

Each training sample payload should contain:

- page image or region image
- canonical target text
- sample metadata
- optional auxiliary annotations

## Git-Visible Companion Artifacts

For each published bundle, keep in git:

- bundle summary table
- source composition table
- slice coverage summary
- publish ledger entry
