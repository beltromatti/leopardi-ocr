# Pretraining Objectives

Date locked: 2026-04-08

Leopardi-S0 should learn through one main objective and several cheap but high-value auxiliary objectives.

## Main Objective

### Canonical autoregressive decoding

Train the writer to predict canonical Markdown plus LaTeX.

Why:

- this is the product contract
- small models benefit from being trained on the exact output form they must later emit

This main objective should not remain uniform across all target tokens.
Formula spans and table spans should carry explicit extra weight because they dominate the public frontier gap.

## Auxiliary Objectives

### Block type prediction

Why:

- reduces decode entropy
- stabilizes planner behavior

### Block length bucket prediction

Why:

- helps the decoder budget itself before writing

### Specialist hint prediction

Why:

- teaches the planner when math, table, handwriting, or chart pressure is present

### Block box regression

Why:

- creates a useful alignment pressure between visual structure and written structure

### Planner confidence calibration

Why:

- makes later routing and repair decisions more trustworthy
- helps the model expose where cheap decode is unsafe

### Rotation prediction

Why:

- cheap robustness signal that matters for real OCR

### Handwriting difficulty prediction

Why:

- strengthens robustness without requiring a separate model

### Formula token supervision

Why:

- improves formula sensitivity and exact LaTeX recovery

### Table block and span supervision

Why:

- tables remain one of the most important frontier failure modes

## Weighting Policy

Use token loss as the anchor objective.
Use auxiliary losses to steer representation quality, not to dominate the run.

The initial weights live in:

- `configs/pretraining/`
- `src/leopardi/pretraining/losses.py`

The current implementation also supports:

- token-level formula weighting
- token-level table weighting
- sample weighting floors for weak or noisy supervision
- label smoothing for syntax stabilization
