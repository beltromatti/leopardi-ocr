# Report Contract

Date locked: 2026-04-08

## Directory Key

Every report package should be keyed by:

- protocol version
- experiment id
- hardware tag
- decode mode

## Minimum Contents

- run metadata
- run manifest pointer
- artifact pointers
- dataset bundle summary
- scorecards
- latency card
- failure-slice summary
- competitor comparison table when applicable
- evidence notes

## Immutability Policy

- `draft` reports may change
- `candidate` reports should change only for corrections
- `promoted` and `frozen` reports must be immutable
