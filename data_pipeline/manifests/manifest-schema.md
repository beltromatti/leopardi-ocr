# Manifest Schema

Date locked: 2026-04-08

This file defines the schema hierarchy the Leopardi data engine should emit.

## Entity Levels

### Source record

Fields:

- `source_id`
- `source_name`
- `version`
- `access_status`
- `license_notes`
- `raw_modality`

### Document record

Fields:

- `doc_id`
- `source_id`
- external identifiers such as DOI, arXiv id, or PMC id
- `doc_hash`
- language hints
- high-level difficulty hints

### Page record

Fields:

- `page_id`
- `doc_id`
- page index
- render parameters
- orientation metadata
- page hash

### Sample record

Fields:

- `sample_id`
- `page_id` or region identity
- `data_class`
- `task_family`
- `target_type`
- `canonical_target_hash`
- `difficulty_tier`
- `slice_tags`
- `split_assignment`
- `bundle_assignment`
- `transform_recipe`
- `publish_artifact_uri`

## Bulk Manifest Format

Persistent machine-readable manifests should default to:

- Parquet for large tables
- JSONL only when needed for line-oriented inspection

Repo-side summaries should default to:

- compact CSV
- markdown cards

## Schema Stability Rule

Once a promoted bundle uses a schema version, that version becomes append-only for that bundle lineage.
