# Model References

Date locked: 2026-04-08

This file anchors the model-layer design to the research already synthesized in the repo.

## Internal Research Anchors

- `docs/architecture.md`
- `docs/research/frontier-synthesis-2026-04.md`
- `docs/research/foundation-model-frontier-2026-04.md`
- `docs/research/efficient-training-frontier-2026-04.md`
- `docs/research/compression-and-efficiency-frontier-2026-04.md`
- `docs/research/leopardi-blueprint-inputs-2026-04.md`
- `docs/research/open-source-codebase-audit-2026-04.md`

## External Frontier References

- `Nougat`
  - compact document-to-markup modeling reference
- `olmOCR`
  - benchmark, training, and parsing-stack reference
- `PaddleOCR-VL`
  - strong compact document parser reference
- `GLM-OCR`
  - compact efficiency-oriented OCR-VL reference
- `FireRed-OCR`
  - structure-validity and RL-oriented parser reference
- `UniMERNet`
  - specialist formula-recognition ceiling
- `Qwen2.5-VL`, `InternVL3`, `SmolVLM`, `Libra`
  - broader VLM architecture frontier relevant to compact multimodal design

## Codebase Anchors In `external/`

- `external/competitors/olmocr/`
- `external/competitors/glm-ocr/`
- `external/competitors/paddleocr/`
- `external/competitors/firered-ocr/`
- `external/frontier-runtime/vllm/`
- `external/frontier-runtime/sglang/`
- `external/frontier-structured/xgrammar/`
- `external/frontier-structured/llguidance/`

The model layer should stay compatible with the actual runtime and deployment stack implied by those codebases.
