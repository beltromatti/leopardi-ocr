# Logging Policy

Date locked: 2026-04-08

Leopardi logging must be useful under SSH, not verbose by habit.

## Logging Layers

### 1. `console.log`

Human-readable append-only log.

Use for:

- phase transitions
- checkpoint saves
- evaluation milestones
- warnings requiring intervention

### 2. `events.ndjson`

Structured append-only event stream.

Use for:

- machine-readable state transitions
- metric snapshots
- artifact publication events
- sync failures

### 3. `heartbeat.json`

Small mutable file for quick status inspection.

Use for:

- current step
- latest metrics
- last save timestamp
- last sync timestamp

This file exists so an SSH agent can reconnect and inspect progress immediately without parsing a full log.

## What Not To Persist Externally

These are local-only by default:

- batch-level debug dumps
- very frequent low-level scalar logs
- temporary stack traces already summarized elsewhere

## Flush Policy

Structured events and console logs should flush often enough that disconnects do not hide recent progress.

Recommended rule:

- flush on milestone events immediately
- flush heartbeats on a short interval

## Logging Goal

The logging system should answer three questions quickly:

1. is the run alive
2. what is it doing now
3. what is the latest durable artifact
