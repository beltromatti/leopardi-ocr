# Frontier Codebase Stack Audit

Date locked: 2026-04-07

This document complements the OCR competitor audit by covering the general-purpose open-source stack that most directly shapes the Leopardi design space.

These are not competitors in the product sense. They are the codebases that determine what kinds of models, post-training loops, structured outputs, and deployment targets are practically achievable.

## Why This Audit Matters

The OCR-VL frontier is no longer determined only by better page parsers.

It is increasingly determined by:

- whether the model can be served efficiently
- whether RLVR can be run at acceptable throughput
- whether quantization can preserve accuracy
- whether constrained decoding can enforce exact outputs with low latency

## Runtime Stack

### `vllm`

Submodule:

- `external/frontier-runtime/vllm`

Why it matters:

- still the default open serving reference for high-throughput LLM and VLM serving
- deeply relevant for paged KV, speculative decode, quantization, and multimodal support

Most important signal from the current repo state:

- active development on quantized KV-cache paths and modern serving features

Leopardi implication:

- the baseline deployment story should work in `vLLM`, even if `SGLang` becomes faster for some regimes

### `sglang`

Submodule:

- `external/frontier-runtime/sglang`

Why it matters:

- unusually strong momentum on frontier-model support, structured outputs, and large-scale serving
- increasingly important rollout backend for RL and VLM work

Most important signal from the current repo state:

- explicit investment in prefill-decode disaggregation, expert parallelism, and structured outputs

Leopardi implication:

- `SGLang` should be treated as a first-class target rather than a later port

### `flashinfer`

Submodule:

- `external/frontier-runtime/flashinfer`

Why it matters:

- kernel layer where much of the real inference advantage now lives
- supports paged and ragged KV, MLA, FP8/FP4, MoE, and serving-oriented fused ops

Most important signal from the current repo state:

- rapid support for current-generation GPUs and MoE/TensorRT-LLM aligned kernels

Leopardi implication:

- if Leopardi eventually uses compact MoE or aggressive quantization, this layer becomes directly relevant

## Training and Post-Training Stack

### `verl`

Submodule:

- `external/frontier-training/verl`

Why it matters:

- strongest open post-training stack for RLVR-style workflows with modern rollout engines
- already supports VLM RL and verifiable rewards

Most important signal from the current repo state:

- strong integration story with `vLLM`, `SGLang`, `FSDP2`, and large-model post-training

Leopardi implication:

- this is the reference stack for structural-reward training of Markdown and LaTeX outputs

### `liger-kernel`

Submodule:

- `external/frontier-training/liger-kernel`

Why it matters:

- one of the most concrete open levers for reducing training memory and increasing throughput
- especially relevant for post-training and VLM SFT

Most important signal from the current repo state:

- support spans base training, post-training losses, and VLM SFT

Leopardi implication:

- use it as a lever for training compact models harder and cheaper rather than defaulting to bigger hardware budgets

### `torchtitan`

Submodule:

- `external/frontier-training/torchtitan`

Why it matters:

- clean reference implementation for PyTorch-native large-scale training
- keeps the training stack understandable and extensible

Most important signal from the current repo state:

- strong support for `FSDP2`, context parallelism, float8, and distributed checkpointing

Leopardi implication:

- useful as a pretraining and systems reference even if final post-training happens elsewhere

## Compression Stack

### `torchao`

Submodule:

- `external/frontier-compression/torchao`

Why it matters:

- strongest PyTorch-native optimization path spanning training-to-serving
- directly relevant to float8 training, QAT, edge deployment, and `vLLM` compatibility

Most important signal from the current repo state:

- clear integration story across training, quantized serving, and mobile/on-device paths

Leopardi implication:

- TorchAO should be one of the first places to look for quantization-aware Leopardi variants

### `llm-compressor`

Submodule:

- `external/frontier-compression/llm-compressor`

Why it matters:

- practical deployment-oriented compression stack closely aligned to `vLLM`
- handles KV-cache and attention quantization, not just weight-only quantization

Most important signal from the current repo state:

- explicit support for VLM quantization, MoE quantization, and fine-grained KV-cache pathways

Leopardi implication:

- critical for the “small, fast, accurate” deployment path once a baseline model exists

## Structured Decoding Stack

### `xgrammar`

Submodule:

- `external/frontier-structured/xgrammar`

Why it matters:

- widely integrated into `vLLM`, `SGLang`, and `TensorRT-LLM`
- structured generation is directly relevant to strict Markdown and LaTeX emission

Most important signal from the current repo state:

- structured generation is now treated as core serving infrastructure, not an application-side wrapper

Leopardi implication:

- grammar-aware decoding should be an explicit design branch for exact Markdown output

### `llguidance`

Submodule:

- `external/frontier-structured/llguidance`

Why it matters:

- strong benchmark posture on constrained decoding and low-overhead token masking
- already integrated in major runtimes and product surfaces

Most important signal from the current repo state:

- benchmark-driven emphasis on low-overhead masking and grammar coverage

Leopardi implication:

- worth studying as a candidate decoding or repair backend, especially if Markdown AST or LaTeX fragments can be enforced locally

## Key Audit Conclusions

### 1. Leopardi needs a two-runtime worldview

The research and code both support the same conclusion:

- `vLLM` is the safest serving baseline
- `SGLang` may become the faster path for frontier deployments and RL rollouts

### 2. RLVR is now practical enough to be first-class

Because `verl` already integrates with modern runtimes and supports verifiable rewards, Leopardi can plan for:

- Markdown validity rewards
- table-structure rewards
- LaTeX syntax rewards
- latency-aware reward shaping

### 3. Compression should be designed in, not bolted on

`TorchAO` and `llm-compressor` make it clear that:

- training precision
- serving precision
- KV quantization
- QAT

must all be considered earlier in the lifecycle

### 4. Structured decoding is no longer optional research

The combination of `xgrammar` and `llguidance` changes the picture:

- exact output validity is no longer just a finetuning problem
- grammar engines are now fast enough to be part of product architecture

## What This Changes For The Blueprint

The final Leopardi blueprint should assume:

- dual runtime targets: `vLLM` and `SGLang`
- post-training via a `verl`-compatible stack
- quantization-aware development with `TorchAO` and `llm-compressor`
- explicit structured decoding experiments using `xgrammar` and `llguidance`
