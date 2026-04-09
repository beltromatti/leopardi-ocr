# Transform Families

Date locked: 2026-04-08

This file defines the approved transform families for `synthetic_exact` data.

## Label-Preserving Families

The canonical target must remain unchanged.

### Geometry

- arbitrary page rotation
- local block rotation
- mild perspective distortion
- scale changes
- border crops that do not remove semantic content

### Scan Noise

- blur
- JPEG artifacts
- contrast shifts
- grayscale variation
- copier noise
- shadow and illumination effects
- bleed-through style overlays

### Surface Markup

- highlight marks
- stamps
- underlines
- redaction bars only when they do not hide supervised text
- handwritten annotations whose text is separately known and inserted into the target when semantically present

## Exact Compositional Families

The canonical target is regenerated from known components.

### Printed document composition

Combine:

- text blocks from exact corpora
- tables from table corpora
- formulas from exact or specialist formula corpora
- chart regions from graphics corpora

Constraint:

- every inserted component must have known content and provenance

### Handwriting-over-print composition

Combine:

- printed exact page
- known handwriting snippet or note

Constraint:

- if the handwriting is intended to be parsed, it must be inserted into the canonical target explicitly
- if it is pure distractor noise, it must be tagged as such

## Text Corruption Families

Date added: 2026-04-09

Source: MiniCPM-V 4.5 (2025) demonstrated that dynamic text region corruption
during pre-training forces the vision encoder to develop stronger OCR features
and the decoder to rely on linguistic context, eliminating dependence on
perfect visual input.

### Text Region Corruption

The canonical target remains unchanged. The model must infer corrupted text
from surrounding context and layout.

Applied during P2 (10% corruption rate) and P3 (20-40% corruption rate):

- heavy Gaussian blur on selected text regions (sigma 5-15)
- block noise fill replacing text with random pixels
- contrast reduction making text regions near-invisible
- partial character erosion simulating damaged scans

Detection of text regions:

- use line density map from the page canonicalizer
- regions with density above threshold are candidate text areas
- randomly select 10-40% of candidate regions per page

Constraint:

- the canonical target is NEVER modified
- corruption is visual-only — the model learns to reconstruct from context
- each corrupted region should be tagged in metadata for analysis

### Resolution Degradation

Applied during P3 only:

- downsample page image to 72-96 DPI then upsample back
- JPEG compression at quality 20-50
- simulate phone camera: perspective warp + motion blur + noise

## Forbidden Transform Families

- any transform that silently changes text content without regenerating the target
- semantic cutouts that remove content but keep the old target
- teacher-generated hallucinated page compositions with no exact provenance
