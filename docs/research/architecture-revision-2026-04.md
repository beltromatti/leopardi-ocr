# Architecture Revision: Pretrained Backbone Integration

Date locked: 2026-04-09

This document records the research basis for upgrading `Leopardi-S0` from an
`~86M` fully-trained-from-scratch model to a `~200M` model built on top of
pretrained vision and language components, and the coherent scaling path to
`Leopardi-S1 ~600M`.

## Why The Revision

The previous `~86M` blueprint invested `~30%` of its parameter budget on novel
components (structural latent bottleneck, block planner, layout side-map
encoder) while leaving only `~9.4M` for the vision encoder and `~50M` for the
decoder, both trained from scratch.

Every competitive system in the `100M`–`600M` class uses pretrained
components:

| System | Params | Pretrained Encoder | Pretrained Decoder |
|--------|--------|-------------------|--------------------|
| SmolDocling | 256M | SigLIP-base 93M | SmolLM2-135M |
| Nougat | 350M | Swin-B ~88M (ImageNet) | mBART ~262M |
| GOT-OCR 2.0 | 580M | VitDet-base ~80M | Qwen-0.5B ~500M |
| GLM-OCR | 0.9B | CogViT ~400M | GLM-0.5B ~500M |
| SmolVLM-256M | 256M | SigLIP-base 93M | SmolLM2-135M |

Research evidence for pretrained components at small scale:

- SmolDocling (256M, pretrained both) beats Qwen2.5-VL-7B on DocLayNet
- GOT-OCR 2.0 (580M, pretrained both) beats models 100x its size
- ICCV 2025 Oral (Scaling Laws for Native Multimodal Models): early-fusion
  with pretrained encoders outperforms at low parameter counts
- PP-OCRv5 (2026): even 5M-parameter OCR benefits from pretrained features
- SmolVLM paper: freezing the vision encoder during initial training stages
  is standard and effective

## Chosen Pretrained Vision Encoder: SigLIP2-base-patch16-NaFlex

Source: Google (arXiv:2502.14786, February 2025)

Model ID: `google/siglip2-base-patch16-naflex`

### Why SigLIP2-base-NaFlex

1. **NaFlex excels on OCR and document tasks**: the NaFlex variant processes
   documents at their native aspect ratio, minimizing distortion. The paper
   explicitly notes NaFlex superiority on OCR-based retrieval tasks.

2. **92.93M parameters**: ViT-B/16 backbone. Fits within the 200M budget while
   leaving meaningful capacity for the decoder and novel components.

3. **State-of-the-art pretrained features**: SigLIP2 outperforms SigLIP,
   DFN, and OpenCLIP at all model scales. It includes improvements from
   decoder-based pretraining, self-distillation, and masked prediction.

4. **Multilingual**: trained on multilingual image-text data, providing a
   foundation for future multilingual extension.

5. **Widely adopted**: used in Granite Vision 3.3, SmolVLM2, and other
   production VLMs as of 2026.

6. **Scales cleanly**: ViT-B (92.93M) for S0, ViT-L (303M) or So400m (400M)
   for S1 with the same architecture family.

### Architecture Details (ViT-B/16)

- Architecture: Vision Transformer Base
- Hidden size: 768
- Layers: 12
- Attention heads: 12
- Patch size: 16
- Parameters: 92.93M (vision tower only)
- NaFlex: supports variable sequence lengths and native aspect ratios
- 2D learned positional embeddings

### Integration Plan

- Freeze bottom 8 layers during P2 core multimodal pretraining
- Fine-tune top 4 layers from the start of P2
- Unfreeze progressively during P3 and finetuning stages
- Use pixel shuffle (2×2→1) to reduce visual token count by 4×
- Project from 768 to internal hidden dimension (576) via MLP

## Chosen Pretrained Language Decoder Base: SmolLM2-135M Architecture with Qwen3 Innovations

The decoder combines two sources:

### 1. Architectural innovations from Qwen3

Source: Qwen Team (arXiv:2505.09388, May 2025)

Key features adopted for the Leopardi writer decoder:

- **RoPE** (Rotary Position Embeddings) with theta=1000000
- **GQA** (Grouped Query Attention): 9 query heads, 3 KV heads at 576 hidden
- **SwiGLU** activation in FFN
- **RMSNorm** with pre-normalization
- **QK-Norm** for training stability
- **No QKV bias** (following Qwen3 finding)

These are strictly superior to the previous design (learned absolute position
embeddings, standard attention, GELU, LayerNorm).

Qwen3-0.6B config for reference:
- hidden_size: 1024, layers: 28, heads: 16Q/8KV, intermediate: 3072
- vocab: 151936, max_pos: 40960, head_dim: 128, RoPE theta: 1000000

### 2. Weight initialization from SmolLM2-135M

Source: HuggingFace (2025)

SmolLM2-135M provides pretrained self-attention and FFN weights trained on
2 trillion tokens of curated text data.

SmolLM2-135M config:
- hidden_size: 576, layers: 30, heads: 9Q/3KV, intermediate: 1536
- vocab: 49152

### Integration strategy

The Leopardi writer decoder operates at hidden_size=576 with 12 layers.
Initialization from SmolLM2:

1. Select 12 layers distributed across SmolLM2's 30 layers (indices
   0, 3, 5, 8, 11, 13, 16, 18, 21, 24, 26, 29)
2. Copy self-attention and FFN weight matrices at native 576 hidden size
   without down-projecting the language prior
3. Add cross-attention layers (randomly initialized) for conditioning on
   structural latents, planner outputs, and layout tokens
4. Replace token embeddings with Leopardi's domain-specific tokenizer
   (randomly initialized, quickly learned during P1 text warmup)

This gives the decoder strong pretrained language patterns while allowing
full customization of the output vocabulary and cross-modal conditioning.

## Parameter Budget: `Leopardi-S0` at ~200M

| Component | Params | Source |
|-----------|--------|--------|
| SigLIP2-base-NaFlex vision encoder | 92.93M | Pretrained (Google) |
| Pixel shuffle + Projection MLP (768→576) | ~1.77M | Random init |
| Structural Latent Bottleneck (3 layers, 192 latents, 576) | ~13.39M | Random init |
| Block Planner (3 layers, 64 queries, 576) | ~13.33M | Random init |
| Layout Side-Map Encoder | ~0.3M | Random init |
| Writer Decoder (12 layers, 576, GQA 9Q/3KV, SwiGLU, RoPE) | ~77.36M | SmolLM2 init |
| MTP heads (horizon=2) | ~0.5M | Random init |
| Auxiliary heads | ~0.01M | Random init |
| **Total** | **~199M** | |

### Trainable parameter phases

- **P1 text warmup**: decoder only (~77M trainable)
- **P2 core multimodal**: decoder + novel components + top 4 SigLIP layers
  (~106M native plus the low-LR unfrozen vision subset)
- **P3 hard cases**: all parameters unfrozen (~199M trainable)
- **F0–F3 finetuning**: all parameters except explicit repair/RLVR freezes (~199M trainable)

## Scaling Path: `Leopardi-S1` at ~600M

| Component | S0 (200M) | S1 (600M) | Scale factor |
|-----------|-----------|-----------|-------------|
| Vision encoder | SigLIP2-base 92.93M | SigLIP2-base 92.93M | 1× (same) |
| Internal hidden | 576 | 960 | 1.67× |
| Projection | 768→576 | 768→960 | scale connector |
| Latent bottleneck layers | 3 | 6 | 2× |
| Latent count | 192 | 384 | 2× |
| Planner layers | 3 | 5 | 1.67× |
| Planner queries | 64 | 112 | 1.75× |
| Decoder layers | 12 | 27 | 2.25× |
| Decoder init | SmolLM2-135M | SmolLM2-360M | next tier |
| MTP horizon | 2 | 3 | 1.5× |
| **Total** | **~199M** | **~606.7M** | 3.05× |

The S1 decoder initialized from SmolLM2-360M (hidden=960, 32 layers):
- Select 27 layers at native 960 hidden size
- Same cross-attention addition strategy as S0
- Much stronger pretrained language priors

The vision encoder stays SigLIP2-base in both S0 and S1. This ensures
perfect comparability. All quality improvement at S1 scale comes from
increased decoder, bottleneck, and planner capacity, not from changing
the visual front-end. If S1 needs stronger vision, SigLIP2-Large (303M)
can be swapped in as a controlled ablation.

## Key Architectural Improvements Over Previous Design

### 1. Pretrained vision encoder (SigLIP2-base-NaFlex)

Previous: 9.4M ConvNeXt from scratch.
New: 92.93M SigLIP2 pretrained on billions of image-text pairs.
Expected impact: +5–10 points on document parsing benchmarks.

### 2. Modern decoder architecture (RoPE, GQA, SwiGLU, RMSNorm)

Previous: learned absolute position embeddings (max 2048), standard MHA,
GELU, LayerNorm.
New: RoPE (unlimited extrapolation), GQA (efficient KV cache), SwiGLU
(better approximation), RMSNorm (faster, stabler), QK-Norm.
Expected impact: better length generalization, faster training, stabler
convergence.

### 3. Pretrained decoder initialization (SmolLM2)

Previous: trained from scratch.
New: self-attention and FFN weights from SmolLM2 (2T tokens of pretraining).
Expected impact: faster convergence, better output fluency, reduced P1
warmup time needed.

### 4. Larger decoder (12 layers at hidden 576 vs 11 at hidden 448)

Previous: 11 layers at hidden=448, ~50.5M total (30.8M in blocks).
New: 12 layers at hidden=576, ~77M total with cross-attention for
richer conditioning and GQA for efficient attention.
Expected impact: comparable or better quality per parameter due to modern
architecture and pretrained initialization.

### 5. Pretrained-aligned hidden dimension (576)

Previous: 448 (non-standard).
New: 576 (native SmolLM2-135M width with 9Q/3KV heads).
Expected impact: cleaner pretrained transfer and less initializer distortion.

### 6. Increased max sequence length (4096)

Previous: 2048.
New: 4096 with RoPE (and extrapolatable beyond).
Expected impact: handles complex pages with many formula blocks and tables.

## RTX 5090 Optimization Notes

The RTX 5090 (Blackwell, SM 12.0, compute capability 12.0) requires:

- **CUDA 12.8+** and PyTorch with SM 12.0 support (torch 2.9+)
- **FlashAttention**: as of April 2026, flash-attn support for SM 12.0 is
  still evolving. Fallback: `torch.nn.functional.scaled_dot_product_attention`
  with `torch.compile` provides near-FlashAttention performance.
- **torch.compile**: the primary optimization path for RTX 5090 training.
  All modules should be compile-friendly (no dynamic control flow in
  forward pass, standard ops).
- **bf16 training**: Blackwell has strong bf16 tensor core support.
- **Liger-Kernel**: fused cross-entropy, RMSNorm, SwiGLU, and RoPE kernels
  available in `external/frontier-training/liger-kernel/`.

The model code must be written with `torch.compile` compatibility as the
primary constraint. This means:
- No Python-level branching on tensor values in forward()
- Standard PyTorch ops throughout
- F.scaled_dot_product_attention instead of manual attention

## Sources

- SigLIP 2 paper: https://arxiv.org/abs/2502.14786
- SigLIP 2 HuggingFace blog: https://huggingface.co/blog/siglip2
- SigLIP2-base-NaFlex model: https://huggingface.co/google/siglip2-base-patch16-naflex
- SmolLM2-135M: https://huggingface.co/HuggingFaceTB/SmolLM2-135M
- SmolLM2-360M: https://huggingface.co/HuggingFaceTB/SmolLM2-360M
- Qwen3-0.6B config: https://huggingface.co/Qwen/Qwen3-0.6B
- Qwen3 Technical Report: https://arxiv.org/abs/2505.09388
- SmolDocling: https://arxiv.org/abs/2503.11576
- SmolVLM paper: https://arxiv.org/abs/2504.05299
- Liger-Kernel: https://github.com/linkedin/Liger-Kernel
- RTX 5090 flash-attn status: https://github.com/Dao-AILab/flash-attention/issues/1638
