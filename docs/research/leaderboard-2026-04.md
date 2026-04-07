# Leaderboard Snapshot

Date locked: 2026-04-07

This file is a research synthesis, not an official leaderboard submission. It combines public benchmark data, official repo claims, and vendor-authored reports with evidence grades.

## Executive Takeaways

- `PaddleOCR-VL-1.5` is the strongest publicly documented all-around single-page document parsing competitor right now.
- `HunyuanOCR` may be stronger still on multilingual and wild parsing, but most of the strongest numbers are vendor-authored rather than benchmark-maintainer reported.
- `dots.mocr` is the most strategically important competitor for Leopardi because it is strong on both document parsing and structured graphics parsing.
- `FireRed-OCR` is the clearest example of syntax-constrained RL explicitly targeting structural validity.
- `GLM-OCR` and `MonkeyOCR-pro-1.2B` are the most relevant efficiency-oriented compact competitors.
- `olmOCR 2`, `Infinity-Parser`, and `Chandra` are critical competitors for English PDF-to-Markdown quality on difficult PDFs.

## Public Benchmark Leaders

### OmniDocBench v1.5

Top reported page parsing scores from benchmark-maintainer tables and current model reports:

1. PaddleOCR-VL-1.5: 94.5 overall in paper abstract, evidence `B`
2. HunyuanOCR: 94.10 overall on vendor-authored report, evidence `C`
3. FireRed-OCR: 92.94 overall in paper abstract, evidence `B`
4. PaddleOCR-VL: 92.86 overall on OmniDocBench repo, evidence `A`
5. MinerU2.5: 90.67 overall on OmniDocBench repo, evidence `A`
6. OpenDoc-0.1B: 90.57 overall on OpenOCR repo, evidence `B`
7. Qwen3-VL-235B-A22B-Instruct: 89.15 overall on OmniDocBench repo, evidence `A`
8. MonkeyOCR-pro-3B: 88.85 overall on OmniDocBench repo, evidence `A`
9. OCRVerse: 88.56 overall on OmniDocBench repo, evidence `A`
10. dots.ocr: 88.41 overall on OmniDocBench repo, evidence `A`

### olmOCR-Bench

Current leaders on difficult English PDF linearization:

1. dots.mocr: 83.9 overall on dots.mocr repo, evidence `B`
2. Chandra OCR 0.1.0: 83.1 overall on olmOCR repo table, evidence `A`
3. Infinity-Parser 7B: 82.5 overall on olmOCR repo table, evidence `A/B`
4. olmOCR v0.4.0 / olmOCR 2 family: 82.4 overall on olmOCR repo table, evidence `A`
5. PaddleOCR-VL: 80.0 overall on olmOCR repo table, evidence `A`
6. Marker 1.10.1: 76.1 overall on olmOCR repo table, evidence `A`
7. MonkeyOCR-pro-3B: 75.8 overall on MonkeyOCR repo, evidence `B`
8. DeepSeek-OCR: 75.7 overall on olmOCR repo table, evidence `A`
9. MinerU 2.5.4: 75.2 overall on olmOCR repo table, evidence `A`
10. Mistral OCR API: 72.0 overall on olmOCR repo table, evidence `A`

### IDP OCR Leaderboard

Best current OCR-only snapshot for handwriting and rotation:

1. gemini-2.5-pro-preview-03-25: 81.18 average OCR, evidence `A`
2. gemini-2.0-flash: 80.05 average OCR, evidence `A`
3. gemini-2.5-flash-preview-04-17: 78.90 average OCR, evidence `A`
4. gpt-4.1-2025-04-14: 75.64 average OCR, evidence `A`
5. gpt-4o-2024-11-20: 74.91 average OCR, evidence `A`
6. o4-mini-2025-04-16: 72.82 average OCR, evidence `A`
7. qwen2.5-vl-72b-instruct: 69.61 average OCR, evidence `A`
8. claude-3.7-sonnet: 69.19 average OCR, evidence `A`
9. mistral-medium-3: 69.05 average OCR, evidence `A`
10. InternVL3-38B-Instruct: 66.31 average OCR, evidence `A`

## Provisional Leopardi Ranking

This is a weighted research ranking for Leopardi's target product:

- one PDF page
- arbitrary rotation
- mixed print and handwriting
- tables, charts, headings, formulas
- Markdown output with LaTeX
- accuracy and latency both matter

### Rank 1: PaddleOCR-VL-1.5

Why it leads:

- best currently published all-around page parsing score among compact public models
- explicit Real5 robustness push for physical distortions
- 0.9B size is excellent for the current frontier

Main uncertainty:

- public latency numbers are less standardized than accuracy numbers

### Rank 2: HunyuanOCR

Why it matters:

- strongest multilingual and wild-document evidence in current vendor report
- 1B parameter footprint
- end-to-end parsing plus spotting, IE, translation

Main uncertainty:

- the strongest headline numbers are not yet benchmark-maintainer or third-party reproduced

### Rank 3: dots.mocr

Why it matters:

- best strategic fit against Leopardi's chart and diagram ambitions
- strongest open signal on graphics-as-first-class parsing
- 83.9 on olmOCR-Bench is elite

Main uncertainty:

- fewer neutral public benchmark tables than PaddleOCR-VL and OmniDocBench entries

### Rank 4: FireRed-OCR

Why it matters:

- 92.94 on OmniDocBench v1.5
- explicit data factory plus format-constrained RL
- directly attacks structural hallucination

### Rank 5: GLM-OCR

Why it matters:

- compact 0.9B
- multi-token prediction explicitly targets throughput
- pragmatic two-stage layout plus region recognition design

### Rank 6: MonkeyOCR family

Why it matters:

- best disclosed speed/accuracy tradeoff among the stronger open systems
- explicit pages-per-second numbers on several GPUs
- mature modular SRR design

### Rank 7: MinerU2.5

Why it matters:

- strong public benchmark numbers
- decoupled high-resolution parsing design is directly relevant to hard pages

### Rank 8: OCRVerse

Why it matters:

- holistic text-centric plus vision-centric OCR
- two-stage SFT-RL multi-domain training

### Rank 9: Infinity-Parser / Chandra / olmOCR 2

Why they matter:

- extremely serious English PDF-to-Markdown competitors
- valuable for stress-testing Markdown fidelity and difficult layouts

### Rank 10: OpenDoc-0.1B / UniRec-0.1B

Why they matter:

- tiny model footprint with real benchmark credibility
- strong signal that small specialist systems can still beat larger VLMs in narrow regimes

## Speed and Footprint Leaders

### Best disclosed speed evidence

1. MonkeyOCR-pro-1.2B: about 1.76 to 2.42 pages/s in VLM OCR mode on RTX 4090 depending on page count, evidence `B`
2. MonkeyOCR-pro-3B: about 1.32 to 1.41 pages/s in VLM OCR mode on RTX 4090, evidence `B`
3. GLM-OCR: no public pages/s table in the sources reviewed, but multi-token prediction is a direct throughput optimization, evidence `B`
4. PaddleOCR-VL / PaddleOCR-VL-1.5: compact 0.9B and repeatedly described as fast, but public pages/s figures were not found in the first pass, evidence `B`

### Best compact models

1. OpenDoc-0.1B / UniRec-0.1B: 0.1B class
2. PaddleOCR-VL / PaddleOCR-VL-1.5: 0.9B
3. GLM-OCR: 0.9B
4. HunyuanOCR: 1B
5. MinerU2.5 and MonkeyOCR-pro-1.2B: 1.2B

## What Leopardi Must Beat

To credibly claim frontier status, Leopardi needs to beat at least:

1. PaddleOCR-VL-1.5 on OmniDocBench-style page parsing
2. dots.mocr or Chandra on difficult PDF-to-Markdown fidelity
3. HunyuanOCR on multilingual and photographed documents
4. MonkeyOCR-pro-1.2B on disclosed speed/accuracy efficiency
5. UniMERNet / Mathpix / GPT-4o class systems on formula-to-LaTeX exactness
