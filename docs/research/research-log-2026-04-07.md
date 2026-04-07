# Research Log

Date: 2026-04-07

## Goal

Map the current frontier competitors for Leopardi across:

- page parsing accuracy
- PDF-to-Markdown quality
- formula fidelity
- handwriting and rotation robustness
- model size and deployment footprint
- latency and speed where publicly disclosed

## Working Conclusions

### Frontier open specialized systems

- PaddleOCR-VL-1.5
- HunyuanOCR
- FireRed-OCR
- GLM-OCR
- dots.mocr
- MonkeyOCR v1.5 / pro family
- MinerU2.5
- OCRVerse
- olmOCR 2
- OpenDoc-0.1B

### Production-commercial references

- Mistral OCR
- Mathpix
- Gemini family
- OpenAI family

### Benchmarks that actually matter for Leopardi

- OmniDocBench v1.5
- Real5-OmniDocBench
- olmOCR-Bench
- IDP OCR leaderboard
- MDPBench
- formula/table subsets from OmniDocBench and related specialist datasets

## Gaps Found

- public latency reporting is still much weaker than accuracy reporting
- many vendor papers do not publish hardware-normalized pages-per-second
- several strong PDF-to-Markdown systems have weak public architecture disclosure
- wild-photo and multilingual evaluation is still less standardized than clean PDF parsing

## Immediate Research Tasks After This Pass

1. fetch full PaddleOCR-VL-1.5, GLM-OCR, FireRed-OCR, and HunyuanOCR PDFs and build finer latency/ablation notes
2. build a benchmark ingestion layer for OmniDocBench, olmOCR-Bench, and IDP OCR
3. define Leopardi's own markdown-validity and latex-validity verifiers
4. create a reproducible competitor evaluation harness on a fixed hardware target

