# Mosaic Archive Packing

Mosaic Archive Packing is a reversible pre-transform for lossless compression. It was added because the repo needed more than "pick the smallest codec"; it needed an actual data-layout transform that may expose additional structure before entropy coding.

## Intuition
Traditional compressors see a byte stream. Mosaic first reorders the stream into byte planes across fixed-width words, then optionally applies a prefix-XOR residual to each plane. This can make adjacent bytes more predictable for the downstream codec.

## Encoding steps
Given a byte string `x` and a word size `w`:
1. Pad `x` to a multiple of `w`.
2. Split into `w` byte planes.
3. Reorder planes using a permutation.
4. Optionally apply prefix-XOR residual coding on each plane.
5. Compress the transformed stream with a standard codec.

## Decoding steps
1. Decompress the stored stream.
2. Undo the residual transform.
3. Reassemble the planes into words.
4. Truncate the original padding.

## Why this is useful
A lot of model artefacts are structured but not obviously textual:
- safetensors fragments
- binary blobs
- repeated numeric patterns
- mixed-entropy payloads with local correlations

Mosaic is a cheap reversible transform that can make those structures more obvious to a generic codec.

## Exactness
Mosaic is lossless because it only permutes and re-encodes bytes reversibly. No information is discarded.

## Current implementation
The implementation lives in `file 'middleout_lattice/mosaic.py'` and is integrated into the archive selection logic.

## Caveat
This is still not magic. It is a better front-end for lossless coding, not a promise that every file will compress well.
