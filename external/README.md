# External Codebases

This directory vendors the most important external codebases for Leopardi research.

It is split into two families:

- `competitors/`: direct OCR and document-parsing competitors
- `frontier-*`: general-purpose runtime, training, compression, and structured-generation stacks that shape the broader frontier

## Why These Repos Exist

Leopardi should not be designed from papers alone.

The current frontier is increasingly determined by:

- serving runtime behavior
- kernel quality
- post-training infrastructure
- quantization pathways
- structured decoding engines

Those ingredients are often clearer in code than in model papers.

## Main Categories

- `competitors/`: OCR-VL competitors and specialist references
- `frontier-runtime/`: serving engines and inference kernels
- `frontier-training/`: training and post-training frameworks
- `frontier-compression/`: quantization and deployment tooling
- `frontier-structured/`: constrained decoding engines relevant to exact Markdown and LaTeX output

## Research Links

- `docs/research/open-source-codebase-audit-2026-04.md`
- `docs/research/frontier-codebase-stack-2026-04.md`
- `docs/research/final-refinement-2026-04.md`
