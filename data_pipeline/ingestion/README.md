# Ingestion

Date locked: 2026-04-08

Ingestion is the front door of the Leopardi data system.

Its job is to:

- identify valid public sources
- acquire only the needed assets
- pair source documents with trustworthy targets
- emit candidate manifests for curation

It is not allowed to guess labels or silently rewrite targets.

## Ingestion Priorities

Priority order is strict for `Leopardi-S0`.

### Priority A: exact pair foundations

- arXiv source plus PDF
- PMC Open Access XML plus PDF

### Priority B: specialist public supervision

- PubLayNet
- DocLayNet
- PubTables-1M
- SciTSR
- CROHME
- MathWriting
- Im2LaTeX-100K
- IAM
- Bentham
- READ 2016
- FUNSD
- CORD
- SROIE
- ChartQA
- PlotQA

### Priority C: optional extensions after verification

- MonkeyDoc and other newly released public corpora
- CHURRO-DS and related historical-text extensions
- layout-rich corpora referenced by competitors but requiring license or schema review

FinTabNet family has now been promoted out of this bucket and pinned to a
public aligned parquet release for automated ingestion.

## Files

- `source-registry.csv`
  - the authoritative source inventory and role map
- `source-endpoints.csv`
  - automated probe policy and acquisition-surface coverage for each source
- `source-priority.md`
  - why each core source matters and when to ingest it
- `acquisition-policy.md`
  - download, cache, and purge rules for real machines
- `pairing-and-targets.md`
  - how sources become canonical targets
- `research-watchlist.md`
  - high-interest sources that remain conditional until verified
