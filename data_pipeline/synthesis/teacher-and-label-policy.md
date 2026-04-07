# Teacher And Label Policy

Date locked: 2026-04-08

Leopardi may use strong external tools, but only in controlled roles.

## Allowed Teacher Roles

### Mining

Use frontier parsers or document tools to:

- find likely hard cases
- surface disagreement slices
- rank candidate pages for review

### Weak auxiliary labeling

Allowed only in explicitly tagged `weak_aux` pools.

Use for:

- rough block hints
- candidate reading-order hints
- candidate region proposals

### Validator assistance

Use tools to:

- compare outputs
- flag malformed Markdown
- detect likely missing blocks

## Forbidden Teacher Roles

- replacing source-native targets in exact pools
- silently upgrading weak labels into exact bundles
- using benchmark-evaluated teacher outputs as training truth

## Practical Competitive Insight

Recent competitors show that data engines matter, but their papers also imply a risk:

- geometry-aware synthesis is useful
- high-quality Markdown normalization is useful
- disagreement mining is useful

The mistake would be to let teacher predictions become hidden truth.

Leopardi must not make that mistake.
