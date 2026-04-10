# Data Configs

Store dataset mixture and target-format presets here.

These configs should point to named data families and split versions, not ad hoc file paths.

Current control-plane entry points:

- `document_parser_core.yaml`
  - shared product-scope data contract
- `s0_exact_core_build.yaml`
  - exact-pair-first build for the `~150M` research vehicle
- `s0_full_frontier_build.yaml`
  - rare full-stack build for the `~150M` frontier-scale data family (`~10.3M` target samples)
- `s1_full_frontier_build.yaml`
  - scaled full-stack build for the final `~500M` family
