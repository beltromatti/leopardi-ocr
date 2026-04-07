# Ingestion

Focus areas:

- born-digital PDF plus source-target pairs
- scanned and degraded documents, including page-level crops and document-native PDFs
- handwriting-heavy pages
- math- and table-dense pages

Every page should be versioned by manifest entry with source URI, license, split, and target format.

All ingestion outputs should register into:

- `data_pipeline/manifests/`
- `data_pipeline/registry/`
