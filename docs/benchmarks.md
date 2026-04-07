# Benchmark Plan

## North-Star Metrics

- page-level exactness on normalized Markdown
- document-level exactness on normalized Markdown
- math exactness on normalized LaTeX
- structural fidelity for tables and reading order
- latency per page and per document at fixed hardware budget
- throughput under batch and document-serving workloads

## Required Benchmark Families

- general document parsing: OmniDocBench and successors
- PDF-to-Markdown extraction: olmOCR Bench and related long-tail PDF suites
- challenging real-world pages: Real5-OmniDocBench
- table structure: PubTables-1M, FinTabNet, SciTSR
- math OCR: CROHME, Im2LaTeX-style sets, scientific PDF math subsets
- handwriting: IAM, Bentham, and document-level handwritten pages

## Evaluation Rules

- Report both accuracy and latency on the same hardware.
- Track both page-level and document-level metrics because public benchmarks are often page-based while products are document-based.
- Separate clean born-digital PDFs from degraded scans.
- Track easy, medium, and hard routing buckets.

## Baseline Grid

- classic OCR pipeline: detector + recognizer + layout model + formatter
- OCR-free VLM parser
- hybrid parser with expert routing
- commercial API comparison when licensing allows
