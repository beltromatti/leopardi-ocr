# Protocol: competitor_reproduction_v1

Date locked: 2026-04-08

## Purpose

This protocol governs how Leopardi compares itself to competitors in a fair and reproducible way.

## Scope

Primary competitor set:

- `PaddleOCR-VL-1.5`
- `HunyuanOCR`
- `dots.mocr`
- `FireRed-OCR`
- `GLM-OCR`
- `MonkeyOCR`
- `MinerU2.5`
- `OCRVerse`
- `olmOCR 2`
- `Infinity-Parser`
- `Chandra`
- `OpenDoc-0.1B`

Specialist references:

- `UniMERNet`
- `Mathpix`

## Comparison Rules

1. Prefer official open implementations when available.
2. Use official public benchmark tables when direct local reproduction is not yet possible.
3. Keep reproduced numbers separate from literature numbers.
4. Mark every competitor entry with evidence grade.
5. Never silently compare different decode modes or hardware classes.

## Required Output Tables

- reproduced open baselines
- literature baselines
- commercial API references
- size-normalized ranking
- latency/footprint comparison
