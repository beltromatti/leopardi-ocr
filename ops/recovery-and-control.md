# Recovery And Control

Date locked: 2026-04-08

Leopardi runs must be operable by an SSH agent that can disconnect and reconnect.

## Heartbeat Rule

Every serious run updates `heartbeat.json` on a short interval.

This is the primary quick-status surface for a reconnecting operator.

## Control Files

Every run should monitor:

- `control/STOP`
- `control/RELOAD`
- `control/NOTE.txt`

### `STOP`

If present:

- finish the current safe unit
- save checkpoint or summary if applicable
- exit cleanly

### `RELOAD`

If present:

- re-read mutable control inputs such as sampling weights or pause flags where supported

### `NOTE.txt`

Use for:

- human or agent notes attached to the run

## Resume Rule

Runs should be resumable from:

- last durable checkpoint
- published bundle state
- latest stage config snapshot
- dataloader or iterator state when the phase streams large published bundles

## Intervention Philosophy

The operator should be able to:

- tail `console.log`
- inspect `heartbeat.json`
- check artifact publication status
- request graceful stop

without needing an interactive notebook or fragile shell state.
