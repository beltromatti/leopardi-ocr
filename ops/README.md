# Operations

Date locked: 2026-04-08

This directory defines the horizontal run contract shared by:

- `data_pipeline/`
- `pretraining/`
- `finetune/`
- `evaluation/`

Its job is to make Leopardi operable on ephemeral rented machines without losing research artifacts.

## Files

- `run-contract.md`
  - canonical run layout and required metadata
- `logging.md`
  - live logging and structured-event policy
- `persistence.md`
  - what survives the machine and where it must be published
- `recovery-and-control.md`
  - heartbeat, stop, reload, and resume policy for SSH agent operation
- `git-surface.md`
  - what belongs in git and what must stay out
- `references.md`
  - primary operational references

The corresponding reusable helpers live in:

- `src/leopardi/ops/`
  - run layout, manifest and heartbeat schemas, file-writing helpers, and control-file probes
