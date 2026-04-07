# Finetuning

Finetuning is split into:

- `sft/`: supervised finetuning for Markdown + LaTeX fidelity
- `rl/`: preference or reward-based optimization for structural validity, formula exactness, and latency-aware tradeoffs

Operationally, finetuning should follow `docs/finetune.md` and use staged configs from `configs/finetune/`.
