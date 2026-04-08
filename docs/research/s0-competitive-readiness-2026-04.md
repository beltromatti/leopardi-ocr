# `S0` Competitive Readiness

Date locked: 2026-04-08

This note records the current evidence-based position of `Leopardi-S0 ~100M` against the 2026 document-parsing frontier.

It is intentionally conservative.
The goal is to identify what the current pipeline can plausibly support, and what it still cannot justify.

## Bottom Line

The current `Leopardi-S0` data and training plan is now defensible as a serious compact-model attempt at the frontier.

It is **not** yet defensible to claim that a `~100M` model will certainly beat:

- `PaddleOCR-VL-1.5`
- `HunyuanOCR`
- `FireRed-OCR`
- `GLM-OCR`

across all public benchmarks before real training and evaluation.

The strongest defensible claim today is narrower:

- `Leopardi-S0` now has a plausible path to becoming one of the strongest compact parsers in the `sub-0.5B` class
- if the exact-first curriculum and later optimization stages work as intended, it can realistically challenge larger parsers on narrow slices such as Markdown fidelity, formula exactness, and table-heavy scientific pages

## Why The Current Plan Is Strong

### 1. The exact-pair core is unusually strong for a compact model

`arXiv + PMC OA` gives Leopardi a cleaner exact core than many OCR stacks that are more heavily reliant on mixed OCR labels or layout-first supervision.

This matters more for a `100M` model than for a `0.9B` model.

### 2. The specialist pool now covers the slices that matter most for product failure

The current pipeline explicitly covers:

- scientific tables
- financial tables
- printed formulas
- handwritten formulas
- handwriting lines and page-like handwriting
- forms and receipts
- chart and plot pages

### 3. The curriculum is now better matched to small-model behavior

The current `S0` plan keeps:

- `P2` exact-first
- `P3` as the first true hard-case escalation stage
- `F0/F1/F2` anchored to exact bundles even when specialist data is active

This is more appropriate for `100M` than the earlier, flatter mixing policy.

## Where The Plan Is Still Weaker Than The Frontier

### 1. Multilingual breadth

Compared with `PaddleOCR-VL-1.5`, the current Leopardi data plan is weaker on multilingual coverage.

This is the largest remaining data-side gap.

### 2. Real-world physical distortion scale

The pipeline now supports deterministic hard-case synthesis, but `PaddleOCR-VL-1.5` explicitly reports `Real5`-style robustness across:

- scanning
- warping
- screen photography
- illumination
- skew

Leopardi still needs the full remote build and later training loop to validate that its synthetic pressure is enough.

### 3. Long-document product behavior

Competitors such as `PaddleOCR-VL-1.5` and `GLM-4.5V` now report increasingly strong long-document claims.

Leopardi's data plan is compatible with document-level assembly, but the claim is not yet earned until the full train/eval loop exists.

## Final Decision

The correct final position for the repo is:

- proceed with the current `S0` data pipeline
- keep the exact-first curriculum locked
- treat multilingual and real-world distortion coverage as the first expansion axes after the first full remote build
- do not promise global `#1` status for `S0` before training
- do expect that the current plan is strong enough to justify the `S0` experiment and, if it works, to support a later `S1 ~500M` scale-up with a credible path to the top tier
