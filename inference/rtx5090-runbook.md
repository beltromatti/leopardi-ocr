# RTX 5090 Inference Runbook

Date locked: 2026-04-08

Leopardi inference on a rented `RTX 5090` should stay simple and honest.

## Primary Policy

- one promoted artifact at a time
- one primary runtime and one explicit fallback runtime
- adaptive `fast` / `standard` / `hard` routing on the same GPU
- keep low-level server logs local only
- persist report cards and summaries outside the machine

## Default Runtime Posture

- `vLLM` first for baseline and broad serving
- `SGLang` second for structured-output-heavy comparison
- use `xgrammar` first, not multiple grammar backends at once

## What To Log

- mode chosen
- route reasons
- repair count
- output token count
- per-page latency
- validator failures
