# Source Priority

Date locked: 2026-04-08

This file explains the order in which Leopardi should spend data-engineering effort.

## Tier A: Must-Have Before First Serious Training

### arXiv source plus PDF

Why first:

- strongest open source of scientific exact pairs
- native formulas
- dense tables
- captions, references, footnotes, and multi-column layouts

Leopardi role:

- tokenizer corpus
- text warmup corpus
- core multimodal pretraining
- exact SFT

### PMC Open Access XML plus PDF

Why first:

- high-quality structured XML supervision
- complementary article styles beyond arXiv
- important biomedical tables and references

Leopardi role:

- core multimodal pretraining
- document assembly supervision
- exact SFT

## Tier B: Must-Have Specialist Sources

### Layout

- `PubLayNet`
- `DocLayNet`

Why:

- stabilize region semantics, reading order hints, and layout coverage beyond exact-pair corpora

### Tables

- `PubTables-1M`
- `SciTSR`

Why:

- table topology remains one of the highest-error parsing areas
- small models need explicit table pressure

### Formulas

- `CROHME`
- `MathWriting`
- `Im2LaTeX-100K`

Why:

- exact LaTeX capability is a differentiator
- isolated formula data remains useful even if the final task is full-page parsing

### Handwriting

- `IAM`
- `Bentham`
- `READ 2016`

Why:

- competitor stacks are still weak on handwriting mixed with layout
- Leopardi needs real handwriting sources before synthesis

### Business and noisy documents

- `FUNSD`
- `CORD`
- `SROIE`

Why:

- receipts, forms, and noisy business pages expose structure failures early

### Graphics and charts

- `ChartQA`
- `PlotQA`

Why:

- graphics-aware parsing is still underdeveloped outside a few frontier systems

## Tier C: Optional After Verification

### FinTabNet family

Value:

- dense financial tables and header complexity

Condition:

- use only if licensing and split hygiene are verified

### MonkeyDoc

Value:

- public data-generation pipeline details from a serious recent competitor

Condition:

- inspect release terms, schema quality, and overlap risk before inclusion

### CHURRO-DS and related historical-text corpora

Value:

- stronger historical handwriting coverage

Condition:

- verify access terms and target compatibility before promoting beyond research status
