# Data Pipeline Revision: Scaling to Frontier Quality and Quantity

Date locked: 2026-04-09

This document records the evidence-based revision of the Leopardi data pipeline
to close the remaining quantity gap against competitors while maintaining and
extending the quality advantage.

## Why This Revision

The previous S0 pipeline was designed for a first research loop:

- arXiv: 2,500 documents (~20K pages)
- PMC OA: 2,000 documents (~16K pages)
- Total exact: ~36K pages
- Total with specialists: ~250K-350K samples

Competitor data scales for comparison:

| System | Training Samples | Notes |
|--------|-----------------|-------|
| SmolDocling (256M) | 25.8M | 85% synthetic |
| Nougat (350M) | 8.2M pages | 91% arXiv |
| GOT-OCR 2.0 (580M) | ~15M | 3-stage curriculum |
| MonkeyOCR (3B) | 4.5M | Bilingual |
| OpenDoc/UniRec (0.1B) | 40M | UniRec40M |

Leopardi's quality advantage (exact-pair targets, 8 quality gates, 4+4
stage curriculum) is real. But at 36K exact pages vs Nougat's 8.2M, the
quantity gap is too large even for a quality-first approach.

Research evidence:

- SAIL-VL (2025): logarithmic data scaling — more data always helps
- Beyond Chinchilla (2024): train up to 10K tokens/param for inference-heavy
- PP-OCRv5 (2026): 5M params competitive when data quality is maximal
- CADC (2025): 5% of curated data beats 100% of uncurated — but Leopardi
  already curates, so scaling the curated pool is the correct next step

## New Target Scale for S0

| Data Class | Previous | New Target | Scale Factor |
|-----------|----------|------------|-------------|
| arXiv exact pages | ~20K | ~400K | 20× |
| PMC OA exact pages | ~16K | ~160K | 10× |
| Layout supervision | ~65K | ~65K | 1× (sufficient) |
| Table supervision | ~55K | ~70K | 1.3× |
| Formula supervision | ~60K | ~1.1M | 18× (new sources) |
| Handwriting | ~10K | ~15K | 1.5× |
| Forms/receipts | ~3K | ~5K | 1.7× |
| Charts/plots | ~24K | ~30K | 1.25× |
| European multilingual (NEW) | 0 | ~100K | new |
| Synthetic hard cases | ~50K | ~2M | 40× |
| **Total** | **~300K** | **~4M** | **~13×** |

This brings Leopardi into the same order of magnitude as Nougat (8.2M) and
closer to GOT-OCR (15M), while maintaining strictly higher quality per sample.

## New Sources Added

### Formula Specialist: UniMER-1M

Source: OpenDataLab / wanderkid (arXiv:2404.15254)

- HuggingFace: `wanderkid/UniMER_Dataset`
- Samples: 1,061,791 LaTeX-image pairs
- Composition: Pix2tex (158K) + arXiv (820K) + CROHME (8.8K) + HME100K (74.5K)
- Format: Parquet with `image` + `label` (LaTeX) columns
- License: Apache 2.0
- Quality: high — curated for real-world formula diversity
- CDM metric tooling available in UniMERNet repo

Why it matters:
- 10× larger than Im2LaTeX-100K
- Covers both printed and handwritten formulas
- Includes complex multi-line expressions
- Directly compatible with Leopardi's formula specialist path

Integration:
- Ingest via HF parquet streaming (same pattern as mathwriting, im2latex_100k)
- Target: `$latex_label$` for inline, `$$\nlatex_label\n$$` for display
- Cap at 200K for S0, full 1M for S1

### Rejected: SynthDoG-EN

SynthDoG-EN produces plain Wikipedia text rendered as document images.
Leopardi already has 560K exact-pair pages from arXiv+PMC with structured
Markdown+LaTeX targets — qualitatively far superior. Adding SynthDoG-EN
would dilute the mixture with lower-quality unstructured text targets.

### Multilingual: SynthDoG-European (generated at build time)

Source: generated using the open-source SynthDoG tool from
`github.com/clovaai/donut/synthdog/` with Wikipedia dumps in DE, FR, ES, IT, PT.

Why European and not CJK:
- Leopardi targets PDF→Markdown+LaTeX: a primarily academic/business use case
- The market is anglophone and European, not CJK
- European latin-script languages cost almost zero to the tokenizer
- CJK would require a much larger vocabulary (40960 vocab is insufficient)
- German, French, Spanish, Italian, Portuguese cover the major European
  scientific, legal, and business document languages

Integration:
- Generated as part of the S0 build on the rented machine via
  the production `SynthDoGEuropeanWorker`
- Uses `wikimedia/wikipedia` HF streaming (verified accessible for all 5 languages)
- Renders Wikipedia text as document pages using Pillow + Noto fonts
- Ground truth = the Wikipedia text itself (exact by construction)
- 20K per language × 5 languages = 100K for S0
- 100K per language × 5 = 500K for S1
- `scripts/generate_synthdog_european.py` remains available only as a preview/export utility
- Output in the main build is emitted directly as canonical `page_markdown_projection` samples
- Tag: `synthetic`, `multilingual`, `european`, `{language_code}`
- Estimated disk: ~18 GB for S0 (100K × ~180KB avg per image)

### Rejected: OCR-MLT-50M

Probed via script on 2026-04-09. Dataset has NULL images (text-only labels
with no paired image data) and suspicious placeholder text content.
Not suitable for Leopardi training. Removed from pipeline.

### Formula Specialist: CMER-3M (watchlist → conditional)

Source: Bai et al. (AAAI 2026, arXiv:2512.13731)

- GitHub: `https://github.com/Baitlo/CMER`
- Samples: 3.1M complex formula image-LaTeX pairs
- Subset of MER-17M (17.7M)
- Quality: high — emphasis on complex multi-line expressions

Integration decision:
- Add to research watchlist pending release verification
- If download confirmed accessible: promote to `trusted_aux` for S1
- Not required for S0 (UniMER-1M is sufficient and verified accessible)

## Text Corruption Pre-Training

Source: MiniCPM-V 4.5 (2025)

MiniCPM-V 4.5 introduced "text corruption" during pre-training: dynamically
corrupting text regions in document images with varying noise levels. This
forces the model to learn text recognition from visual features (where text
is corrupted/hidden) AND knowledge reasoning from linguistic context (where
text is visible). The technique eliminates reliance on external OCR parsers.

Integration for Leopardi:

Add a new transform family to `synthesis/transform-families.md`:

### Text Region Corruption

Applied to exact-pair and trusted-aux samples during P2 and P3:

1. **Detect text regions** using cheap heuristics (line density map from canonicalizer)
2. **Randomly corrupt 10-40% of text regions** per page with:
   - Gaussian blur (strong, sigma 5-15)
   - Block noise fill
   - Contrast reduction to near-background
3. **Keep the canonical target unchanged** — the model must infer corrupted
   text from linguistic context and surrounding layout

Why it matters:
- Forces the vision encoder to develop stronger local OCR features
- Forces the decoder to develop stronger language model priors
- Reduces dependence on perfect visual input
- MiniCPM-V 4.5 showed this is a key training innovation

Implementation:
- Add to synthetic transform families as `text_corruption`
- Apply during P2 (10% corruption rate) and P3 (20-40% rate)
- The target stays exact — this is a label-preserving transform

## Scaled arXiv and PMC

### arXiv: 2,500 → 50,000 documents

Changes:
- `from_date` in ArxivSourceWorker: `2008-01-01` (wider range)
- `DEFAULT_SOURCE_LIMITS_S0["arxiv_source_pdf"]`: 50,000
- `max_pages_per_document`: 12 (from 8)
- Expected yield: ~50K docs × ~8 usable pages = ~400K exact pages

Disk impact:
- Each doc: ~5-10MB raw (PDF + source), ~2MB canonical
- Total raw peak: ~250-500GB (streamed, purged per-source)
- Published canonical: ~80-100GB

### PMC OA: 2,000 → 20,000 documents

Changes:
- `DEFAULT_SOURCE_LIMITS_S0["pmc_oa_pdf_xml"]`: 20,000
- `max_pages_per_document`: 12 (from 8)
- Expected yield: ~20K docs × ~8 usable pages = ~160K exact pages

## Updated Mixture Targets for S0

### Pretrain stages

| Stage | exact_pair % | synthetic_exact % | trusted_aux % | multilingual % | weak_aux % |
|-------|-------------|-------------------|---------------|---------------|-----------|
| p1_text_warmup | 100 | 0 | 0 | 0 | 0 |
| p2_exact_core | 60 | 10 (text corruption) | 25 | 5 | 0 |
| p3_hardcases | 30 | 40 | 15 | 10 | 5 |

### Finetune stages

| Stage | exact_pair % | synthetic_exact % | trusted_aux % | multilingual % | weak_aux % |
|-------|-------------|-------------------|---------------|---------------|-----------|
| f0_general_sft | 85 | 0 | 10 | 5 | 0 |
| f1_specialist_sft | 30 | 30 | 30 | 5 | 5 |
| f2_repair_sft | 45 | 40 | 10 | 5 | 0 |
| f3_rlvr | 55 | 20 | 15 | 5 | 5 |

## Scaling Path to S1

For S1 (500M), scale all sources proportionally:

| Source | S0 Limit | S1 Limit |
|--------|---------|---------|
| arXiv | 50,000 docs | 200,000 docs |
| PMC OA | 20,000 docs | 80,000 docs |
| UniMER-1M | 200,000 | 1,000,000 |
| SynthDoG-European (DE/FR/ES/IT/PT) | 100K (20K each) | 500K (100K each) |
| CMER-3M | 0 (watchlist) | 500K (if verified) |
| Synthetic hard cases | 2M | 5-8M |
| **Total S0** | **~3.7M** | |
| **Total S1** | | **~15-20M** |

## Storage Estimate for S0 Full Build

The pipeline streams and purges — raw sources are never accumulated on disk.
The peak occurs when the largest bundle (p2_exact_core) accumulates its
shard output before publishing to HuggingFace.

| Phase | Disk Need | Notes |
|-------|----------|-------|
| p2_exact_core shards (peak) | ~247 GB | arXiv 160GB + PMC 64GB + SynthDoG 23GB |
| p2_structural_aux shards | ~20 GB | After p2_core is published and purged |
| p3_hardcases shards | ~28 GB | After p2_aux is published and purged |
| Working cache + OS | ~15 GB | Constant overhead |
| **Peak total** | **~262 GB** | During p2_exact_core only |
| **Recommended free disk** | **400 GB** | With safety margin |

Published dataset total on HuggingFace: **~376 GB** across all bundles
(includes overlap — same arXiv/PMC pages appear in pretrain and finetune
bundles).

## Sources

- UniMER-1M: https://huggingface.co/datasets/wanderkid/UniMER_Dataset
- UniMERNet paper: https://arxiv.org/abs/2404.15254
- CMER-3M/MER-17M paper: https://arxiv.org/abs/2512.13731
- CMER GitHub: https://github.com/Baitlo/CMER
- SynthDoG tool (Donut): https://github.com/clovaai/donut
- OCR-MLT-50M: REJECTED — probed 2026-04-09, images are NULL, text is placeholder
- MiniCPM-V 4.5 text corruption: https://huggingface.co/openbmb/MiniCPM-V-4_5
- SAIL-VL scaling laws: https://arxiv.org/abs/2501.05952
- Beyond Chinchilla: https://arxiv.org/abs/2401.00448
