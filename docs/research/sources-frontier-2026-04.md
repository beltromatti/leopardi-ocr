# Sources: General Frontier

Date locked: 2026-04-07

This registry underpins the broader frontier pass beyond OCR-specific competitors.

Evidence grades:

- `A`: direct primary source with strong relevance to Leopardi
- `B`: primary source with good directional value but weaker directness, maturity, or deployment signal

## Architecture and Foundation Models

- DeepSeek-V2. MoE plus MLA efficiency signal. Evidence `A`. https://arxiv.org/abs/2405.04434
- DeepSeek-V3 Technical Report. MoE plus MLA plus MTP plus FP8 co-design. Evidence `A`. https://arxiv.org/abs/2412.19437
- Jamba. Hybrid Transformer-Mamba language model. Evidence `A`. https://arxiv.org/abs/2403.19887
- Zamba. Compact SSM-transformer hybrid. Evidence `A`. https://arxiv.org/abs/2405.16712
- Kimi Linear. Linear-attention architecture with strong long-context efficiency claims. Evidence `B`. https://arxiv.org/abs/2510.26692
- OpenELM. Open compact LM with architecture-aware parameter allocation. Evidence `A`. https://arxiv.org/abs/2404.14619
- Qwen2.5-VL Technical Report. Strong open VLM baseline with native-resolution relevance. Evidence `A`. https://arxiv.org/abs/2502.13923
- InternVL3. Open multimodal training and test-time recipe signal. Evidence `A`. https://arxiv.org/abs/2504.10479
- Libra. Decoupled vision system with routed visual expert. Evidence `A`. https://arxiv.org/abs/2405.10140
- Qwen2.5-Omni Technical Report. Modality-aware multimodal architecture. Evidence `A`. https://arxiv.org/abs/2503.20215
- SmolVLM. Compact open VLM family for low-memory deployment. Evidence `A`. https://arxiv.org/abs/2504.05299
- MindVL. Efficient multimodal training system with native-resolution angle. Evidence `B`. https://arxiv.org/abs/2509.11662
- Firebolt-VL. Efficient VLM with linear-time decoder direction. Evidence `B`. https://arxiv.org/abs/2604.04579
- Empirical Recipes for Efficient and Compact Vision-Language Models. Deployment-shaped compact VLM paper. Evidence `B`. https://arxiv.org/abs/2603.16987
- LinguDistill. Selective cross-modal distillation for language recovery in VLMs. Evidence `B`. https://arxiv.org/abs/2604.00829

## Data, Distillation, and Training

- JEST. Joint example selection for multimodal learning. Evidence `A`. https://arxiv.org/abs/2406.17711
- Molmo and PixMo. Open-data quality and open-weights VLM recipe. Evidence `A`. https://arxiv.org/abs/2409.17146
- Visual Program Distillation. Distills tools and programmatic reasoning into VLMs. Evidence `A`. https://arxiv.org/abs/2312.03052
- DeepSeek-R1. Reinforcement-learning-first reasoning signal. Evidence `A`. https://arxiv.org/abs/2501.12948
- ViPER. Self-evolving visual perception in VLMs. Evidence `B`. https://arxiv.org/abs/2510.24285
- HybridFlow. Flexible and efficient RLHF framework. Evidence `A`. https://arxiv.org/abs/2409.19256
- DCPO. Dynamic clipping policy optimization. Evidence `B`. https://arxiv.org/abs/2509.02333
- DaGRPO. Distinctiveness-aware GRPO direction. Evidence `B`. https://arxiv.org/abs/2512.06337
- DeepVideo-R1. Difficulty-aware GRPO for video. Evidence `B`. https://arxiv.org/abs/2506.07464
- UI-Venus. Efficient reward design for screenshot-native VLM agents. Evidence `B`. https://arxiv.org/abs/2508.10833
- Compact Language Models via Pruning and Knowledge Distillation. `Minitron` pruning-and-distillation recipe. Evidence `A`. https://arxiv.org/abs/2407.14679

## Inference and Serving

- PagedAttention / vLLM paper. Evidence `A`. https://arxiv.org/abs/2309.06180
- Medusa. Multi-head speculative decoding. Evidence `A`. https://arxiv.org/abs/2401.10774
- Amphista. Bi-directional multi-head decoding. Evidence `A`. https://arxiv.org/abs/2406.13170
- Alignment-Augmented Speculative Decoding. Evidence `B`. https://arxiv.org/abs/2505.13204
- Mirror Speculative Decoding. Evidence `B`. https://arxiv.org/abs/2510.13161
- FlashInfer. Attention and serving kernels. Evidence `A`. https://arxiv.org/abs/2501.01005
- Jenga. Heterogeneous memory management for LLM serving. Evidence `B`. https://arxiv.org/abs/2503.18292
- Semantic Parallelism. Speculative MoE inference scheduling direction. Evidence `B`. https://arxiv.org/abs/2503.04398
- Occult. Collaborative communication optimization for MoE systems. Evidence `B`. https://arxiv.org/abs/2505.13345
- LVPruning. Language-guided visual token pruning. Evidence `A`. https://arxiv.org/abs/2501.13652
- DivPrune. Diversity-based visual token pruning. Evidence `A`. https://arxiv.org/abs/2503.02175
- ResPrune. Recent text-conditioned subspace reconstruction pruning. Evidence `B`. https://arxiv.org/abs/2603.21105

## Compression and Quantization

- QQQ. Quality-preserving 4-bit quantization. Evidence `A`. https://arxiv.org/abs/2406.09904
- TorchAO. Official PyTorch quantization and sparsity stack. Evidence `A`. https://github.com/pytorch/ao
- LLM Compressor. Official deployment-oriented quantization stack for `vLLM`. Evidence `A`. https://github.com/vllm-project/llm-compressor
- compressed-tensors. Unified compressed tensor formats. Evidence `A`. https://github.com/neuralmagic/compressed-tensors
- HQQ. Flexible low-bit quantization library. Evidence `A`. https://github.com/mobiusml/hqq
- GPTQModel. Cross-hardware GPTQ/AWQ-style quantization toolkit. Evidence `A`. https://github.com/ModelCloud/GPTQModel
- bitsandbytes. Widely used low-bit training and optimizer tooling. Evidence `A`. https://github.com/bitsandbytes-foundation/bitsandbytes

## Runtime and Systems Repositories

- vLLM. Evidence `A`. https://github.com/vllm-project/vllm
- SGLang. Evidence `A`. https://github.com/sgl-project/sglang
- TensorRT-LLM. Evidence `A`. https://github.com/NVIDIA/TensorRT-LLM
- FlashAttention. Evidence `A`. https://github.com/Dao-AILab/flash-attention
- FlashInfer. Evidence `A`. https://github.com/flashinfer-ai/flashinfer
- LMDeploy. Evidence `A`. https://github.com/InternLM/lmdeploy
- DeepEP. Evidence `A`. https://github.com/deepseek-ai/DeepEP
- Mooncake. Evidence `A`. https://github.com/kvcache-ai/Mooncake
- Triton. Evidence `A`. https://github.com/triton-lang/triton
- torchtitan. Evidence `A`. https://github.com/pytorch/torchtitan
- torchtune. Evidence `A`. https://github.com/meta-pytorch/torchtune
- TRL. Evidence `A`. https://github.com/huggingface/trl
- OpenRLHF. Evidence `A`. https://github.com/OpenRLHF/OpenRLHF
- verl. Evidence `A`. https://github.com/volcengine/verl
- LLaMA-Factory. Evidence `A`. https://github.com/hiyouga/LLaMA-Factory
- Liger-Kernel. Evidence `A`. https://github.com/linkedin/Liger-Kernel
- MLX-LM. Evidence `A`. https://github.com/ml-explore/mlx-lm
- MLX-VLM. Evidence `A`. https://github.com/Blaizzy/mlx-vlm
- llama.cpp. Evidence `A`. https://github.com/ggml-org/llama.cpp
- ExecuTorch. Evidence `A`. https://github.com/pytorch/executorch

## Evidence Notes

- This registry mixes peer-reviewed papers, arXiv technical reports, and official project documentation.
- For fast-moving runtime systems, official repositories often contain more current implementation detail than papers.
- All URLs in this registry were resolution-checked during the April 2026 frontier pass.
