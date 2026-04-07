# Acceptance Scorecards

Date locked: 2026-04-08

This file defines how candidate data is accepted into Leopardi bundles.

## Sample-Level Decision

Each sample receives:

- data class
- difficulty tier
- slice tags
- accept, demote, review, or reject decision

## Promotion Rules

### `gold_exact`

Requirements:

- source-native or equivalent exact target authority
- passes all quality gates
- no leakage concerns

### `silver_exact`

Requirements:

- high-confidence canonical target
- minor ambiguity or weaker source authority than `gold_exact`
- safe for training but not the cleanest exact pool

### `synthetic_exact`

Requirements:

- label-preserving or compositionally exact synthesis
- parent truth fully known
- transform recipe version recorded

### `trusted_aux`

Requirements:

- benchmark-grade or corpus-grade auxiliary labels
- clear task role

### `weak_aux`

Requirements:

- useful for mining or auxiliary pressure
- clearly tagged and isolated from exact pools

## Bundle-Level Acceptance

A bundle is promotable only if all are true:

- source composition is declared
- leakage screen passed
- duplicate rate is within expected bounds
- slice coverage is reported
- persistent artifacts were published
- repo-side summary tables were updated
