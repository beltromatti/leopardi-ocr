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

## Forbidden Transform Families

- any transform that silently changes text content without regenerating the target
- semantic cutouts that remove content but keep the old target
- teacher-generated hallucinated page compositions with no exact provenance
