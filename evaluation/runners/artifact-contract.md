# Artifact Contract

Date locked: 2026-04-08

Every evaluation run should produce immutable artifacts keyed by:

- experiment id
- protocol version
- hardware tag
- decode mode

## Minimum Artifact Set

- raw predictions archive
- normalized predictions archive
- metric summary table
- latency card
- failure-slice summary
- environment metadata

## Immutability Rule

Once a run is used for promotion or publication, its report artifacts must be treated as immutable.
