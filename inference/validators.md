# Validators

Date locked: 2026-04-08

Inference is only useful if it can tell when the output is structurally broken.

## Mandatory Checks

- balanced code fences
- balanced display and inline math delimiters
- basic table shape consistency
- empty-output and empty-block detection

## Repair Policy

- any hard structural error should trigger repair if the current mode still has repair budget
- warnings should be logged even when repair is skipped
- validation findings must be visible in production and benchmark reports
