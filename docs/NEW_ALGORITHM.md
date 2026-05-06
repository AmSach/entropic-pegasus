# Mosaic Archive Packing

Mosaic is now a hybrid reversible transform. The point is no longer just "reorder bytes". The point is to build a local representation that is easier for a generic lossless codec to compress.

## What it does now
For a binary tensor block, Mosaic can:
- choose a word size adaptively
- order byte planes by measured entropy
- apply one of several reversible residual transforms per plane
- encode plane lengths explicitly when the best transform is not uniform across planes

## Modes
- `natural` and `reversed` plane ordering
- entropy-aware ordering
- `none`, `xor`, `delta`, and `rle` residuals
- automatic per-plane selection

## Why this matters
Old Mosaic was a one-size-fits-all byte permutation. That was too weak to matter against a good whole-file codec.

The new version is more ambitious:
- it adapts to the data rather than assuming one layout
- it tries to flatten local entropy before entropy coding
- it can exploit runs as well as small deltas and XOR regularity

## Exactness
All transforms are reversible. Nothing is lost.

## Best use case
Mosaic is most useful when the data has:
- repeated low-level patterns
- endianness-like structure
- byte-plane asymmetry
- run-length pockets inside binary shards

## Relationship to the tensor pipeline
Mosaic is one candidate inside the tensor-shard stack. The hierarchy is now:
1. raw bytes
2. generic compression
3. Mosaic-transformed compression
4. lattice block packing

## Honest note
Mosaic should not be expected to dominate whole-file compression on every input. Its value is in unlocking extra gains on the blocks where generic compression already did most of the easy work.
