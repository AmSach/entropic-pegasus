# New algorithm: Mosaic Archive Packing

This repo now includes a new exact compression path called Mosaic Archive Packing.

## What it is
Mosaic Archive Packing is a lossless transform that:
- splits a block into fixed-size words
- separates each word into byte planes
- optionally applies XOR residual coding per plane
- lets the archive chooser compare Mosaic against ordinary codecs and raw bytes
- stores enough metadata to restore the block exactly

It is not a statistical compressor by itself. It is a structure-extraction transform that can make some byte layouts far more compressible before the normal codecs see them.

## Why this counts as new
The old version only compared generic codecs. Mosaic adds an explicit reversible transform over the byte layout itself.

## How it works
For a block of bytes:
1. Choose a word size, usually 4 or 8.
2. Build byte planes across the words.
3. Optionally apply XOR residual coding to each plane.
4. Concatenate planes into a stream.
5. Let the archive layer compare the transformed stream to raw, zlib, lzma, and bz2.
6. Store the winner and the transform metadata.

## Exact reconstruction
The decoder reverses the steps in the opposite order. Because the transform is reversible and the archive still stores hashes, the result is exact.

## Where it helps
This can help on binary layouts where neighbouring bytes or neighbouring words have structure, for example:
- some tensor shard layouts
- binary blobs with repeated local patterns
- mixed text/binary files with aligned structure

## Where it does not help
If the data is already close to random, Mosaic will usually not beat ordinary codecs. In that case the archive falls back to raw storage.

## Research value
Mosaic gives us a real experimental axis beyond “which off-the-shelf codec is smaller?” It creates a reversible pre-transform that can be studied, benchmarked, and extended into a tensor-aware codec family.
