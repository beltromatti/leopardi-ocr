# Evaluation Protocols

Date locked: 2026-04-08

Protocols are versioned measurement contracts.

They exist so that a result means the same thing across time.

## Protocol Rules

1. A promoted result must name exactly one protocol version.
2. Protocols are append-only once used for a promoted or frozen run.
3. Hardware assumptions are part of the protocol, not an afterthought.
4. Output normalization rules are part of the protocol, not a runner preference.

## Files

- `public_frontier_v1.md`
  - main protocol for external and release-facing claims
- `internal_holdout_v1.md`
  - protocol for internal regression control and promotion review
- `competitor_reproduction_v1.md`
  - protocol for reproducing or comparing against open competitors
- `release_gate_v1.md`
  - minimum gate protocol for promotion to release-candidate status
