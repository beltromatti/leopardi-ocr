# Data Configs

Store dataset mixture and target-format presets here.

These configs should point to named data families and split versions, not ad hoc file paths.

Current control-plane entry points:

- `document_parser_core.yaml`
  - shared product-scope data contract
- `s0_exact_core_build.yaml`
  - exact-pair-first build for the `~100M` research vehicle
- `s0_full_frontier_build.yaml`
  - rare full-stack build once exact core and specialist slices are stable
