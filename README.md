# MiddleOut Lattice

Lossless compression for model files and tensor archives. It is intentionally model-agnostic: if a model can be represented as files, this can test it.

## What this does
- exact round-trip compression for arbitrary bytes
- per-file codec comparison with skip-when-bigger behaviour
- blockwise archive compression with metadata
- a compressed model-store runtime that reads files on demand
- report generation
- SVG + PNG benchmark chart output
- template data for Qwen2.5-0.5B

## Demo and docs
- `file 'docs/TECHNICAL_REPORT.md'`
- `file 'docs/PAPER_OUTLINE.md'`
- `file 'docs/COMPRESSED_MODEL_STORE.md'`
- `file 'data/qwen_template.csv'`
- `file 'paper/main.tex'`
- live demo: https://amsach.github.io/projects/middleout-lattice/demo/
- paper page: https://amsach.github.io/papers/middleout-lattice-paper/

## New compression direction
The current architecture now has a blockwise archive layer that can:
- choose the best codec per block
- leave files uncompressed when that is cheaper
- store a manifest with per-file metadata
- decompress the whole tree back exactly

## Model-store runtime
`CompressedModelStore` can keep a model repo compressed at rest and materialise files on demand.

```python
from pathlib import Path
from middleout_lattice import CompressedModelStore

store = CompressedModelStore.from_source(Path("./model_repo"), Path("./packed"))
print(store.summary())
print(store.read_bytes("config.json")[:80])
```


## New algorithm
- `Mosaic Archive Packing`: a reversible byte-plane + residual transform that can beat generic codecs on structured binary blocks.
- Details: `file 'docs/NEW_ALGORITHM.md'`
- Now integrated into archive selection.

## What this does not promise
- It does not invent a universal free lunch.
- It does not pretend every checkpoint will hit 2x lossless.
- It does keep the decode exact.

## Paper PDF
- `file 'build/main.pdf'`
