# MiddleOut Lattice

Lossless compression for model files and tensor archives. It is intentionally model-agnostic: if a model can be represented as files, this can test it.

## What this does
- exact round-trip compression for arbitrary bytes
- per-file codec comparison
- report generation
- SVG + PNG benchmark chart output
- template data for Qwen2.5-0.5B

## Documentation
- `file 'docs/TECHNICAL_REPORT.md'`
- `file 'docs/PAPER_OUTLINE.md'`
- `file 'data/qwen_template.csv'`

## What this does not promise
- It does not invent a universal free lunch.
- It does not pretend every checkpoint will hit 2x lossless.
- It does keep the decode exact.
