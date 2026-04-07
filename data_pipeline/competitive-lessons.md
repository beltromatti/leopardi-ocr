# Competitive Lessons

Date locked: 2026-04-08

This file distills the most useful data-engine lessons from the current OCR and VLM frontier.

## Useful Lessons To Keep

### FireRed-OCR

Public positioning emphasizes a geometry-plus-semantics data factory and balanced long-tail synthesis.

Leopardi takeaway:

- build hard-case synthesis around geometry and semantic tags together
- do not rely on generic corruption alone

### MonkeyOCR

The project publicly released `MonkeyDoc` and documents a serious multi-source document stack.

Leopardi takeaway:

- treat data generation itself as a first-class competitive asset
- keep synthetic recipes explicit enough to compare and iterate

### GLM family public materials

Recent public materials describe:

- synthetic document rendering
- academic document pairing inspired by Nougat-style alignment
- contamination review

Leopardi takeaway:

- exact academic pairs remain foundational
- contamination control must be built in early

### UniMERNet

The open code path uses WebDataset-friendly multimodal loading patterns.

Leopardi takeaway:

- training payloads should be shard-first, not loose-file-first

## Lessons To Avoid Misreading

### Bigger weak mixtures are not automatically better

Large frontier systems can absorb more noisy data than a `~100M` model.
Leopardi cannot.

### Teacher outputs are not ground truth

Competitor systems often use strong teachers and toolchains.
That does not justify silently treating teacher outputs as exact labels.

### Benchmark-looking synthetic data is not the goal

Leopardi should synthesize product-realistic hard cases, not only benchmark-matching ones.

## Leopardi-Specific Conclusion

The best data engine for Leopardi is:

- exact-pair-first
- synthesis-heavy only where truth is preserved
- shard-first for training readiness
- strict on leakage and weak-label separation
