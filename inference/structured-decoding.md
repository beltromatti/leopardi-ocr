# Structured Decoding

Date locked: 2026-04-08

Leopardi uses structured decoding selectively, not blindly.

## Global Decode

Use low-overhead constraints for:

- block boundary stability
- fenced code closure
- simple table patterns
- math delimiter closure

## Local Repair

Use stronger grammar enforcement only when needed:

- malformed tables
- malformed lists
- invalid LaTeX spans
- broken fenced regions

## Runtime Rule

- `vLLM`
  - default serving baseline
  - structured outputs backend must be logged as part of the decode policy
- `SGLang`
  - strongest path for tighter grammar-driven repair
  - use as explicit runtime condition, not as a hidden optimization

## Backend Rule

- `xgrammar`
  - default backend for current Leopardi work
- `llguidance`
  - keep available for repair-heavy ablations and backend comparison
