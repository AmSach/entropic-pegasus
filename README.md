# MiddleOut Lattice

Lossless compression for model files and tensor archives. It is intentionally model-agnostic: if a model can be represented as files, this can test it.

## What this does
- exact round-trip compression for arbitrary bytes
- per-file codec comparison
- report generation
- SVG benchmark chart output

## What this does not promise
- It does not invent a universal free lunch.
- It does not pretend every checkpoint will hit 2x lossless.
- It does keep the decode exact.
