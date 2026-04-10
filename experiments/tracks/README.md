# Tracks

Tracks partition the Leopardi research program into stable streams.

They prevent unrelated experiments from competing for the same "best model" slot.

## Core Tracks

- `s0-core`: `~100M` general parser
- `s0-repair`: local repair and constrained decoding
- `s0-table`: table-heavy specialist work
- `s0-math`: formula-heavy specialist work
- `s0-runtime`: latency, quantization, and serving experiments
- `s1-core`: `~600M` scale-up, only after `s0-core` is frozen

Each track should have exactly one promoted checkpoint at a time.
