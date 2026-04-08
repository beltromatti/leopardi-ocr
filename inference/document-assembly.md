# Document Assembly

Date locked: 2026-04-08

Leopardi parses pages internally but must emit document-level Markdown externally.

## First-Phase Assembly Policy

- suppress repeated headers when the same first non-empty line repeats across pages
- suppress repeated footers when the same last non-empty line repeats across pages
- keep page-order stable
- emit explicit page-break markers only when a protocol requires them

This is intentionally conservative.

The `~100M` phase should keep document assembly outside the model and make it deterministic.
