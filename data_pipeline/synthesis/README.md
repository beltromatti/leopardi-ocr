# Synthesis

Date locked: 2026-04-08

Synthesis is where Leopardi creates hard cases without destroying truth.

The goal is not to fabricate unrealistic benchmark tricks.
The goal is to reproduce the failure modes that matter in real documents:

- arbitrary rotation
- hard tables
- formula density
- handwriting mixed with layout
- scanned and photographed distortions
- charts and diagrams embedded in text-heavy pages

## Two Synthesis Modes

### 1. Label-preserving transformation

Start from exact data and apply deterministic transforms that keep the canonical target unchanged.

### 2. Exact compositional synthesis

Compose new pages from known public assets whose truth is already available.

This is allowed only when the final target remains fully attributable to known source components.

## Files

- `hardcase-engine.md`
  - the target hard-case families Leopardi should manufacture
- `transform-families.md`
  - approved transform sets and what they are allowed to change
- `teacher-and-label-policy.md`
  - what tools and teacher models may and may not do
