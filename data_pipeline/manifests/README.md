# Data Manifests

Date locked: 2026-04-08

This directory defines the control-plane artifacts that make the data pipeline reproducible.

The manifest layer is what survives after raw data is gone.

## Manifest Principles

1. Every sample must be traceable to a source and transform recipe.
2. Every bundle must be reconstructible from manifests and persistent artifacts.
3. Git stores summaries and schemas; persistent storage stores bulk machine-readable manifests.

## Files

- `manifest-schema.md`
  - required fields and entity hierarchy
- `artifact-layout.md`
  - directory and naming conventions for persistent artifacts
- `publish-and-retention.md`
  - what is retained, where it lives, and when transient data is deleted
