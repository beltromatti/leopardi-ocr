# Leopardi Architecture

## Goal

Beat current OCR and document parsing systems on full-document parsing while retaining strong page-level and region-level accuracy on arbitrary rotation, mixed printed and handwritten text, tables, charts, and mathematics. Output must be Markdown with LaTeX.

Documents are the product object. Pages and regions are internal processing units used for rendering, routing, supervision, benchmarking, and systems optimization.

## Top-1 System Hypothesis

### 1. Page Geometry Canonicalization

- Detect global page orientation and local text-line orientations.
- Canonicalize the raster before heavy decoding.
- Preserve the original coordinate frame so Markdown spans can still be traced back to the source.

### 2. Coarse-to-Fine Visual Encoding

- Run a fast draft encoder over a normalized page render.
- Dynamically allocate higher resolution crops only where the draft path flags uncertainty, dense formulas, handwriting, or tables.
- Keep the token budget small on easy pages to protect latency.

### 3. Mixture of Parsing Experts

- `core_parser`: Markdown-first page transcription.
- `math_expert`: formula segmentation and LaTeX decoding.
- `table_expert`: table structure recovery with Markdown serialization when possible and HTML fallback only internally.
- `chart_expert`: chart title, legend, and axis text extraction.
- `handwriting_expert`: robust line decoding for cursive and noisy scans.

### 4. Structured Decoding

- Decode into a constrained intermediate representation that maps directly to Markdown blocks.
- Enforce block validity during generation so headings, lists, tables, code fences, and display math stay well formed.
- Use normalized math spans for inline `$...$` and display `$$...$$`.

### 5. Verification and Repair

- Run fast verifiers for math syntax, table consistency, reading order, and Markdown validity.
- Trigger repair only on low-confidence spans or invalid structure.
- Prefer localized repair over full-page re-decoding.

## Training Strategy

### Pretraining

- Paired PDF page to Markdown/LaTeX data from born-digital scientific documents.
- Synthetic corruption curriculum for skew, blur, low contrast, occlusion, stamps, annotations, and arbitrary rotation.
- Layout-heavy samples with tables, multi-column structure, equations, captions, and footnotes.

### Finetuning

- SFT on high-quality structured targets.
- Specialist SFT for formulas, tables, and handwriting.
- Preference or reward-based alignment on Markdown validity, formula exactness, and latency-aware accuracy.

## Inference Strategy

- Single-shot fast path for easy pages.
- Draft-then-verify path for medium complexity.
- Expert-routed path for hard pages with formulas, handwriting, or dense tables.

The repo is scaffolded so these strategies can evolve independently without collapsing into a single hard-coded pipeline.
