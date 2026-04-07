# Git Surface

Date locked: 2026-04-08

Git should store the control plane, not the firehose.

## Keep In Git

- experiment registry
- promotion logs
- artifact ledger
- benchmark protocols
- bundle registries
- compact CSV summaries
- report templates
- release cards

## Keep Out Of Git

- checkpoints
- training shards
- large report payloads
- live run directories
- local debug logs

## Why

The repository must remain portable to rented machines and easy to review.
Large mutable runtime state does not belong here.
