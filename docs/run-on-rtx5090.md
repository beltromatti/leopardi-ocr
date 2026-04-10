# RTX 5090 Operator Start

Date locked: 2026-04-08

This file is the single operator entry point for moving Leopardi onto a rented ephemeral machine.

## Disk Envelope For The Full S0 Data Build

For the current split data-pipeline implementation:

- minimum realistic free disk: about `2.0 TB`
- recommended free disk: about `2.5 TB`
- comfortable headroom: `3.0 TB`

This budget assumes the current optimized builder:

- streams parquet-backed HF sources
- processes each source once and dispatches samples into every dependent bundle
- purges source raw/work cache immediately after canonicalization
- publishes completed bundles and drops their local shard copies after verification
- keeps the `~4.5M` derived hard cases out of the first external raw-acquisition wave

## GPU Runtime Packages

The dev machine does not need GPU-serving packages.
The rented `RTX 5090` machine does.

The pinned operator path is:

- `vLLM` via the official quickstart flow with `uv pip install vllm --torch-backend=auto`
- `LLM Compressor` via `pip install llmcompressor`
- `TorchAO` via `pip install torchao`
- `SGLang` via `uv pip install "sglang[all]"`

These commands are encoded in `scripts/bootstrap_rtx5090.sh`.

## What Is Ready Now

The repo is ready as a research and operations control plane:

- architecture, data, pretraining, finetuning, optimization, inference, evaluation, and runtime strategy are locked in docs
- model, data-pipeline, pretraining, finetune, optimization, inference, evaluation, and shared ops layers are importable and tested
- run layout, heartbeat, event logging, control files, and persistence targets are defined
- phase-specific runtime presets exist for data build, pretraining, finetuning, optimization, inference, evaluation, and serving

## What Is Not Implemented Yet

The repo is not yet ready for a full frontier run.

Missing execution layers:

- end-to-end training loop connected to published bundles
- end-to-end finetuning loop connected to published bundles and checkpoints
- optimization export backends that materialize deployable low-bit artifacts
- inference supervisor that boots and measures runtime plans automatically
- benchmark-specific dataset adapters and evaluation supervisors that execute full public protocols automatically

This means the repo is ready for active engineering work on the rented machine, not yet for a final long training campaign.

## First Commands On A Fresh Machine

```bash
./scripts/bootstrap_rtx5090.sh
source .venv/bin/activate
./scripts/smoke_cpu.sh
./scripts/smoke_chain_cpu.sh
python3 -m leopardi.cli --help
python3 -m leopardi.cli doctor
python3 -m leopardi.cli data-pipeline-pretrain-summary configs/runtime/data_build_rtx5090.yaml
python3 -m leopardi.cli data-pipeline-pretrain-materialize leo-s0-data-pretrain-20260410-001 configs/runtime/data_build_rtx5090.yaml --root runs
python3 -m leopardi.cli data-pipeline-pretrain-build leo-s0-data-pretrain-20260410-001 configs/runtime/data_build_rtx5090.yaml --root runs --publish
python3 -m leopardi.cli model-summary configs/model/leopardi_s0.yaml
python3 -m leopardi.cli pretrain-summary configs/pretraining/s0_p2_multimodal_core.yaml configs/runtime/train_rtx5090.yaml
python3 -m leopardi.cli pretrain-materialize leo-s0-p2-20260408-001 configs/pretraining/s0_p2_multimodal_core.yaml configs/runtime/train_rtx5090.yaml configs/model/leopardi_s0.yaml --root runs
python3 -m leopardi.cli finetune-summary configs/finetune/s0_f0_sft.yaml configs/runtime/finetune_rtx5090.yaml
python3 -m leopardi.cli finetune-materialize leo-s0-f0-20260408-001 configs/finetune/s0_f0_sft.yaml configs/runtime/finetune_rtx5090.yaml configs/model/leopardi_s0.yaml --root runs
python3 -m leopardi.cli optimization-summary configs/optimization/s0_o2_vllm_compressed.yaml configs/runtime/optimization_rtx5090.yaml
python3 -m leopardi.cli inference-summary configs/inference/s0_i1_vllm_adaptive.yaml configs/runtime/inference_rtx5090.yaml
python3 -m leopardi.cli evaluation-summary configs/eval/public_frontier.yaml configs/runtime/eval_rtx5090.yaml
python3 -m leopardi.cli evaluation-materialize leo-s0-eval-public-20260408-001 configs/eval/public_frontier.yaml configs/runtime/eval_rtx5090.yaml --root runs
python3 -m leopardi.cli materialize-run-example --root runs
```

## Primary Files To Open First

- `README.md`
- `docs/roadmap.md`
- `docs/experimentation.md`
- `model/README.md`
- `ops/run-contract.md`
- `data_pipeline/README.md`
- `pretraining/README.md`
- `finetune/README.md`
- `optimization/README.md`
- `inference/README.md`
- `evaluation/README.md`

## Recommended First Engineering Sequence

1. Run `data-pipeline-pretrain-build` and publish the pretraining bundles.
2. Run pretraining against those published bundles.
3. On a later machine, run `data-pipeline-finetune-foundation-build`.
4. Publish a failure manifest from the first real model run.
5. Run `data-pipeline-finetune-followup-build`.
6. Implement the optimization export backends and variant validation loop.
7. Implement the inference supervisor and runtime measurement loop.
8. Implement dataset adapters and automated supervisors for `public_frontier_v1` and `internal_holdout_v1`.

## Rule For The Rented Machine

Treat the rented machine as disposable compute, not as durable storage.

Before any long run:

- verify the target HF repositories or other persistent destinations exist
- verify the runtime preset points to the right publication targets
- verify the experiment id, manifest path, and artifact ledger row are prepared
