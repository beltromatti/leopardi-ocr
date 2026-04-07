# Reward Design

Date locked: 2026-04-08

Leopardi uses RL only for rewards that can be checked objectively and cheaply.

## Primary Reward Terms

- Markdown validity
- LaTeX validity
- table structural validity
- reading-order consistency
- normalized edit similarity

## Specialist Reward Terms

- formula exactness
- merged-cell consistency
- header and footer suppression correctness
- chart text and grouping correctness

## Efficiency-Aware Terms

- output length penalty
- latency penalty
- repair budget penalty

## Best-Practice Constraints

Use RL only after stable SFT.

Prefer:

- objective rewards
- short rollout horizons
- dynamic filtering of uninformative groups
- explicit overlong penalties

These choices are aligned with recent open RL practice reflected in `verl` and recipes such as `DAPO`.

## Why RL Is Still Worth It Here

The target is not generic style preference.
It is exact structured parsing under latency pressure.

That is exactly the kind of domain where objective rewards can be unusually useful.
