# Acquisition Policy

Date locked: 2026-04-08

This file defines how Leopardi should acquire data on real rented hardware.

## Acquisition Rules

1. Start with metadata, not bulk downloads.
2. Prefer source-side filtering before local transfer.
3. Download only the subset needed for the current bundle build.
4. Publish processed artifacts before starting the next large acquisition wave.
5. Purge transient raw assets after publication verification unless they are still part of the active build window.

## Local Disk Policy

The first full builds should assume constrained local NVMe and no guarantee of multi-terabyte free space.

Default policy:

- keep source metadata locally
- keep transient raw assets only during active transformation
- keep training-ready shards only until upload verification passes
- delete render caches and raw archives once the persistent artifact is verified

## Download Order

For each source:

1. fetch metadata and license notes
2. score candidate documents or pages
3. materialize only selected items
4. transform to canonical samples
5. publish and checksum
6. purge transient raw data

## Cache Boundaries

Keep separate local areas for:

- `metadata_cache`
- `raw_cache`
- `work_cache`
- `upload_staging`

The policy goal is to make purge decisions easy and safe.

## Persist-Then-Purge Rule

A transient raw asset is deletable once all are true:

- canonical sample artifacts were generated successfully
- manifests were written
- upload to persistent storage completed
- checksums and sample counts match
- exclusion and lineage records were stored in git-controlled metadata or persistent manifests

## What Must Never Be Purged Blindly

- exclusion lists
- sample manifests
- dataset cards
- publish ledgers
- source-to-sample lineage tables
