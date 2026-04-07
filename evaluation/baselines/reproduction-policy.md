# Reproduction Policy

Date locked: 2026-04-08

## Baseline Evidence Tiers

### `reproduced`

Numbers obtained locally under a named Leopardi protocol.

### `official_public`

Numbers taken from official public benchmark tables or papers on public tasks.

### `vendor_authored`

Numbers taken from vendor-controlled reports or benchmark presentations.

### `commercial_reference`

Numbers derived from public benchmarks, official docs, or API runs where exact reproduction is limited.

## Evidence Grade Mapping

Use one evidence grade in scorecards and competitor tables.

- `A`
  - reproduced locally under a pinned Leopardi protocol, or taken from a strong public benchmark table maintained by the benchmark owner with clear setup disclosure
- `B`
  - official paper or official public leaderboard claim on a relevant public task, with mostly clear benchmark conditions
- `C`
  - vendor-authored report, partial reproduction, or otherwise relevant public result with material comparability gaps
- `D`
  - commercial or indirect reference useful for context, but not strong enough for primary scientific claims

Tier-to-grade default:

- `reproduced` -> `A`
- `official_public` -> `A` or `B`
- `vendor_authored` -> `C`
- `commercial_reference` -> `D`

## Rules

1. Do not mix evidence tiers in one table without marking them.
2. Prefer `reproduced` over `official_public` when feasible.
3. Prefer `official_public` over `vendor_authored`.
4. Never report a competitor latency claim without hardware context.
5. If a scorecard uses evidence grades, it should still retain the underlying tier in the report metadata.
