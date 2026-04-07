# Inference Systems Frontier

Date locked: 2026-04-07

This document focuses on inference-time ideas that matter for getting more capability out of fewer active weights and less hardware.

Source registry for this pass: `sources-frontier-2026-04.md`

## Core Serving Frameworks

### vLLM

Why it matters:

- still one of the central open serving runtimes
- `PagedAttention`, continuous batching, quantization support, speculative decoding, multimodal support
- strong ecosystem gravity

### SGLang

Why it matters:

- aggressive optimization for VLMs and modern open models
- strong recent support for DeepSeek-family kernels and disaggregated serving directions
- increasingly important for high-throughput, frontier-model serving

### TensorRT-LLM

Why it matters:

- strongest NVIDIA-centric deployment path for many production systems
- critical if Leopardi ever targets top-end inference on NVIDIA hardware

### LMDeploy

Why it matters:

- serious runtime for compression, serving, and VLM deployment
- strong blocked-KV and quantization posture

### Edge / Local Serving

Important tools:

- `llama.cpp`
- `MLX-LM`
- `MLX-VLM`
- `ExecuTorch`

Why they matter:

- they constrain what "compact enough" really means
- they force the team to think about quantization, memory, and tokenization early

## Kernel and Runtime Layer

### FlashInfer

Why it matters:

- now one of the clearest open kernel engines for attention, GEMM, and MoE serving
- supports paged KV cache, MLA attention, speculative decoding primitives, FP8 and FP4

### FlashAttention

Why it matters:

- still foundational for exact high-speed attention
- remains part of the baseline kernel stack even as serving engines diversify

### Triton

Why it matters:

- open route to custom kernel work
- increasingly central for training and inference kernel innovation

### DeepEP

Why it matters:

- expert-parallel communication is now a first-order systems problem for MoE models

### Mooncake

Why it matters:

- KV-cache-centric prefill/decode disaggregation is one of the clearest systems directions for long-context serving

## Key Inference Algorithms

### PagedAttention

- still foundational for practical high-throughput serving

### Multi-Token Prediction

Important signal:

- `DeepSeek-V3`
- `GLM-OCR`

Why it matters:

- bridges training objective and decode acceleration

### Speculative Decoding

Important signals:

- `Medusa`
- `EAGLE` ecosystem support in runtimes
- `Mirror Speculative Decoding`

Why it matters:

- still one of the strongest general decode-acceleration ideas
- runtime support is now good enough that model design can assume it

### KV Compression and Hierarchical Caching

Important signals:

- `DeepSeek` MLA line
- `Mooncake`
- `LMDeploy`
- `FlashInfer`
- newer KV-cache quantization support in compression/runtime tooling

Why it matters:

- OCR/document systems often have large prefixes, many visual tokens, and variable-length outputs

### Adaptive Visual Token Reduction

Important signals:

- `LVPruning`
- `DivPrune`
- `ResPrune`

Why it matters:

- visual token cost is frequently the bottleneck in VLM serving
- the latest pruning papers are increasingly text-conditioned and training-free or near-training-free

## Leopardi Runtime Guidance

### First-class targets

- `vLLM`
- `SGLang`

### Important optimizations to design for

- paged KV cache
- speculative decoding or MTP-style acceleration
- adaptive visual token budget
- mixed precision and later quantized serving

### Optional but strategically strong

- prefill/decode disaggregation
- KV offload / hierarchy
- MoE-ready communication stack

## Practical Conclusion

If Leopardi is built without explicit runtime assumptions, it will likely lose to smaller but better-served competitors.
