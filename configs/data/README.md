# Data Configs

Store dataset mixture and target-format presets here.

These configs should point to named data families and split versions, not ad hoc file paths.

Current control-plane entry points:

- `document_parser_core.yaml`
  - shared product-scope data contract
- `s0_exact_core_build.yaml`
  - exact-pair-first build for the `~150M` research vehicle
- `s0_pretrain_family_build.yaml`
  - default first-machine build for the full pretraining data family
- `s0_finetune_foundation_build.yaml`
  - default later-machine build for `F0-F1`, reusing published pretraining bundles from HF
- `s0_finetune_followup_build.yaml`
  - default later-machine build for `F2-F3`, consuming a published failure manifest from a real run
- `s0_full_frontier_build.yaml`
  - rare monolithic rebuild for debugging or disaster recovery, not the default operator path
- `s1_full_frontier_build.yaml`
  - scaled full-stack build for the final `~500M` family
