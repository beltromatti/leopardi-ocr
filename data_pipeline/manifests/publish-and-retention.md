# Publish And Retention

Date locked: 2026-04-08

This file defines what must be published and what may be deleted.

## Publish Requirements

Before a bundle is considered real, publish:

- shard payloads
- machine-readable manifests
- data card
- source composition summary
- sample-count summary
- exclusion list summary

## Retention Rules

### Retain persistently

- published training-ready shards
- manifests
- stats
- data cards
- exclusion records
- lineage tables

### Retain in git

- compact CSV summaries
- markdown policy and cards
- operational ledgers

### Delete after verification

- transient raw downloads
- intermediate render caches
- temporary conversion outputs superseded by canonical artifacts

## Verification Before Deletion

No transient purge is allowed before all are true:

- upload completed
- checksums match
- sample counts match
- publish ledger updated
- bundle registry updated

## Reason For This Policy

The rented-machine constraint is real.
The pipeline must optimize for rebuildability, not for hoarding every intermediate forever.
