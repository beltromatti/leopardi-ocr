# Benchmark

Benchmark runners should make it trivial to answer one question:

Can this checkpoint beat the current best accuracy-latency frontier on document parsing while remaining strong on page-level benchmark units?

Keep benchmark execution reproducible and benchmark outputs immutable.

Related subdirectories:

- `datasets/`: benchmark-family adapters
- `runners/`: execution entry points and sweep definitions
- `protocols/`: pinned benchmark protocol versions
