# Compression and Efficiency Frontier

Date locked: 2026-04-07

This document tracks the main ways the field is fitting more useful intelligence into less memory, fewer active parameters, and lower latency.

Source registry for this pass: `sources-frontier-2026-04.md`

## Main Compression Themes

### Quantization Is Now Multi-Layered

The frontier is no longer just "4-bit vs 8-bit".

Important dimensions now include:

- weight-only vs weight-activation quantization
- FP8, INT8, INT4, FP4, MXFP4
- KV-cache quantization
- per-tensor vs per-group vs per-head scaling
- QAT vs PTQ

## Key Libraries and Tooling

### TorchAO

Why it matters:

- PyTorch-native quantization and sparsity
- strong recent support for inference and QAT
- increasingly central for portable open workflows

### LLM Compressor

Why it matters:

- practical deployment-oriented quantization for `vLLM`
- FP8, GPTQ, AWQ, AutoRound, KV quantization, attention quantization

### compressed-tensors

Why it matters:

- unifies storage formats for multiple compression schemes

### bitsandbytes

Why it matters:

- still one of the most practical on-ramps for low-bit finetuning and optimizer-state reduction

### HQQ

Why it matters:

- strong flexible low-bit quantization path with modern PyTorch compatibility

### GPTQ / AWQ / QQQ

Why they matter:

- they remain important reference algorithms and deployment targets
- `QQQ` is especially relevant as a speed-oriented 4-bit direction

## Compactness Beyond Quantization

### Better Parameter Allocation

Best signal:

- `OpenELM`
- `Minitron`

What it suggests:

- layer-wise scaling and architecture-aware parameter placement still matter

### Active-Parameter Efficiency

Best signals:

- `DeepSeek-V2`
- `DeepSeek-V3`
- `Kimi Linear`

What it suggests:

- total parameters matter less than active parameters plus runtime compatibility

### Compact VLM Optimization

Best signals:

- `Empirical Recipes for Efficient and Compact VLMs`
- `Firebolt-VL`
- `SmolVLM` ecosystem and related compact-VLM work

What it suggests:

- smaller VLMs need systems-aware recipes, not just pruning or naive downsizing
- pruning and distillation can be used to manufacture stronger compact students from larger bases

## What Matters Most for Leopardi

### Near-Term

- BF16 and FP8 training/inference awareness
- W4/W8 deployment targets
- KV-cache-aware design
- storage formats that interoperate with modern runtimes

### Mid-Term

- quantization-aware finetuning
- structured sparsity where hardware support is real
- compact student or routed expert variants

### Hard Truth

Compression is no longer a late-stage deployment step. It increasingly feeds back into architecture, training recipe, and runtime choices.
