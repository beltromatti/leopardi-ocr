# Persistence Policy

Date locked: 2026-04-08

Ephemeral machines are allowed.
Ephemeral research artifacts are not.

## Persistence Classes

### Must persist outside the machine

- data bundles
- checkpoints worth keeping
- evaluation report packages
- release cards
- artifact manifests

### Must persist in git

- compact registries
- bundle definitions
- protocol versions
- promotion decisions
- artifact ledger entries

### Local-only by default

- scratch files
- low-level logs
- transient caches
- temporary exports superseded by published artifacts

## Publication Targets

Recommended external targets:

- Hugging Face dataset or model repositories for reusable artifacts
- object-store mirror when volume justifies it

Preferred publication mechanics for large datasets:

- `huggingface_hub` large-folder upload workflow for shard-heavy bundle pushes
- Xet-backed transfer acceleration when available and stable on the rented machine
- manifest-first verification before local purge

## Sync Rule

Every important run must know its target publication locations before it starts.

No run should depend on “we will upload it later” as the primary plan.

## Artifact Classes

- `checkpoint`
- `bundle`
- `report`
- `release_card`
- `summary_table`

Every persisted artifact should have:

- stable URI
- checksum when appropriate
- source experiment id
- phase and stage
- git-visible companion row in the relevant registry or report index
