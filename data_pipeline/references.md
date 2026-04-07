# Data Pipeline References

Date locked: 2026-04-08

This file records the main sources behind the current Leopardi data-engine design.

## Core Data Sources

### Exact pair foundations

- arXiv bulk and source access
- PubMed Central Open Access PDF plus XML

These remain the highest-value public exact-pair foundations for Leopardi.

### Layout and structure

- PubLayNet
- DocLayNet
- PubTables-1M
- SciTSR

### Formula recognition

- CROHME
- MathWriting
- Im2LaTeX-100K

### Handwriting

- IAM
- Bentham
- READ 2016

### Noisy business documents

- FUNSD
- CORD
- SROIE

### Graphics and charts

- ChartQA
- PlotQA

## Competitor And Frontier Inputs

### FireRed-OCR

Why used:

- public emphasis on a geometry-plus-semantics data factory and balanced long-tail synthesis

### MonkeyOCR

Why used:

- public `MonkeyDoc` release and explicit discussion of its data-generation pipeline

### GLM public materials

Why used:

- explicit public discussion of synthetic OCR data, academic document pairing, and contamination checks

### UniMERNet

Why used:

- practical open multimodal shard-first loading patterns

## Data Platform Inputs

### Hugging Face Hub and datasets ecosystem

Why used:

- practical large-artifact publication and downstream distribution
- streaming-friendly dataset workflows

### WebDataset

Why used:

- shard-first multimodal training format with strong precedent in vision-language training

### Megatron-Energon data preparation guidance

Why used:

- confirms shard-first scaling patterns and future growth path if Leopardi outgrows the first single-GPU phase

## Notes

This file is intentionally compact.
Detailed competitive research remains in:

- `docs/research/`
