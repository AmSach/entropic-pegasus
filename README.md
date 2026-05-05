# MiddleOut Lattice

Lossless compression for model files and tensor archives. It is intentionally model-agnostic: if a model can be represented as files, this can test it.

## What this does
- exact round-trip compression for arbitrary bytes
- per-file codec comparison with skip-when-bigger behaviour
- blockwise archive compression with metadata
- report generation
- SVG + PNG benchmark chart output
- template data for Qwen2.5-0.5B

## Documentation
- `file 'docs/TECHNICAL_REPORT.md'`
- `file 'docs/PAPER_OUTLINE.md'`
- `file 'data/qwen_template.csv'`
- `file 'paper/main.tex'`

## New compression direction
The current architecture now has a blockwise archive layer that can:
- choose the best codec per block
- leave files uncompressed when that is cheaper
- store a manifest with per-file metadata
- decompress the whole tree back exactly

## What this does not promise
- It does not invent a universal free lunch.
- It does not pretend every checkpoint will hit 2x lossless.
- It does keep the decode exact.


## Paper PDF
- `file 'build/main.pdf'`
