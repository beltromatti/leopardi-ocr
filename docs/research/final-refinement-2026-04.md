# Final Research Refinement

Date locked: 2026-04-07

This document is the last refinement pass before the final Leopardi blueprint.

It re-reads the current research corpus as a whole:

- OCR and document parsing competitors
- broader VLM and LLM frontier work
- runtime, training, compression, and structured-generation stacks

The purpose is not to reopen the whole search space. It is to identify the remaining gaps that still matter enough to change design decisions.

## What The Research Already Covers Well

### Competitor coverage is now strong enough

We now have:

- benchmark-grounded competitor mapping
- model dossiers
- open-source competitor codebase audit
- unified metrics across fragmented benchmark families

This is enough to define who Leopardi must beat and on which dimensions.

### Broader frontier coverage is also strong enough

We now have:

- compact and efficient LLM/VLM architecture trends
- data and RLVR trends
- inference runtime and kernel trends
- compression and quantization trends

This is enough to avoid building Leopardi from an OCR-only perspective.

### The code-side frontier is now materially represented

We now vendor and can inspect:

- serving stacks
- post-training frameworks
- training-kernel stacks
- quantization stacks
- constrained-decoding engines

This closes one of the largest remaining weaknesses in the research corpus.

## Remaining Gaps That Still Matter

These are not generic “more research is always good” gaps. These are the unresolved areas most likely to affect the blueprint.

### 1. Structured decoding for Markdown plus LaTeX is still underexplored

Why it matters:

- Leopardi does not need generic JSON validity
- it needs exact, low-latency Markdown plus embedded LaTeX

What we now know:

- `xgrammar` and `llguidance` make constrained decoding practically credible
- structured decoding is now integrated in major runtimes

What remains open:

- whether full-document Markdown grammars are too restrictive or slow
- whether block-level or repair-time grammar enforcement is a better fit
- how to combine grammar constraints with expert outputs such as math spans

### 2. Tokenizer and output representation design are still a serious open lever

Why it matters:

- exact Markdown and LaTeX output is partly a tokenizer problem, not only a modeling problem
- poor token boundaries can inflate decode length and error rates

What remains open:

- whether Leopardi should use an off-the-shelf tokenizer unchanged
- whether special handling for tables, list markers, code fences, and LaTeX delimiters is worth the complexity

### 3. Document-level assembly remains less studied than page-level parsing

Why it matters:

- the public benchmark ecosystem is still mostly page-centric
- deployed Leopardi must parse full documents coherently

What remains open:

- cross-page reading order
- repeated header and footer suppression
- carry-over of tables and figures across pages
- document-level confidence and repair orchestration

### 4. The hardest multimodal intersection is still underrepresented in public data

The intersection is:

- handwriting
- layout complexity
- formulas
- graphics
- photographed distortions

Why it matters:

- this is exactly where Leopardi can create separation

What remains open:

- there is still no single open benchmark family that fully covers this intersection
- a Leopardi data engine will still need synthetic and mined hard cases

### 5. Latency reporting remains too weak across the field

Why it matters:

- the user goal is not just best accuracy; it is best accuracy plus speed

What remains open:

- hardware-normalized latency cards
- decode-token normalization
- complexity-tiered page mixes
- runtime-specific structured-decoding overhead measurement

### 6. Sparse OCR-VL remains a live but unresolved question

Why it matters:

- broader research strongly supports active-compute optimization
- OCR-VL research still offers limited open proof that compact sparse models beat compact dense models on this task family

What remains open:

- whether Leopardi v1 should stay dense and compact
- whether optional routing should happen at the system level rather than inside the backbone

## What Is Probably Not Worth More Pre-Blueprint Research

### 1. Chasing every new closed-source multimodal API

Useful as a market signal, but not as a design template.

### 2. Over-optimizing for one benchmark family

The research is already clear that no single benchmark captures Leopardi’s whole target.

### 3. Reading more generic LLM scaling papers without OCR or systems relevance

At this point the bottleneck is integration, not a lack of abstract scaling-law awareness.

## Most Important New Conclusion From This Refinement Pass

Leopardi is not just an OCR-VL model problem.

It is a co-design problem across:

- compact multimodal modeling
- verified data generation
- RLVR
- runtime selection
- quantization
- structured decoding
- document-level orchestration

That is the most important correction produced by the full research pass.

## Pre-Blueprint Recommendations

Before freezing the final blueprint, Leopardi should explicitly decide:

1. whether grammar enforcement is decode-time, repair-time, or hybrid
2. whether the first model family is dense, routed, or hybrid-system
3. which runtime is primary and which is secondary
4. which quantization target is mandatory from day one
5. which document-level orchestration responsibilities stay outside the core model

## Companion Documents

- `frontier-synthesis-2026-04.md`
- `frontier-codebase-stack-2026-04.md`
- `leopardi-blueprint-inputs-2026-04.md`
- `unified-metrics.md`
