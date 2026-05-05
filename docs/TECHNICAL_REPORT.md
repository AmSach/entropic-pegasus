# Technical report: MiddleOut Lattice

## What we built
MiddleOut Lattice is a lossless compression prototype for model files and tensor archives. The important point is that it is **exact**: whatever goes in must come back byte-for-byte.

The repo was shaped around a simple reality:
- KV-cache compression is not whole-model compression.
- If you want whole-model lossless reduction, you need to compress the **files that define the model**.
- The most honest first target is the on-disk model assets and metadata files used by a model repository.

## What changed in the second phase
### 1. Added a blockwise archive codec
The new archive layer lives in `middleout_lattice/archive.py` and does something more useful than a single file compressor:
- it splits files into blocks
- it tries multiple codecs on every block
- it keeps the smallest encoding for that block
- it leaves the file uncompressed when compression would make it bigger
- it stores a manifest with all metadata needed to reverse the process

This is the key new behaviour the user asked for: **compress separately per file, remember the metadata, and skip compression when a file does not benefit**.

### 2. Added a model-agnostic directory archive
The archive layer can compress a whole tree of files into a mirror structure plus `manifest.json`.
That means model repos can be handled as collections of files rather than as one monolithic blob.

### 3. Kept the codec exact
The archive still performs checksum verification, size checks, and exact byte reconstruction.
There is no approximation step here.

## How the blockwise archive works
For each file:
1. split the file into fixed-size blocks
2. for each block, evaluate several codecs
3. include a raw block option so the system can choose not to compress a block
4. store per-block metadata: codec, raw size, stored size, SHA-256
5. if the final archive is not smaller than the original file, keep the original file as raw

That last rule matters. A lossless compressor should never force a worse representation just because it is trying to feel clever.

## Why this is a step toward ultra compression
This is not yet a magic universal LLM compressor. It is an architecture that creates the right shape for one:
- file-local decisions
- block-local decisions
- exact metadata
- decompression at read time
- model-tree level manifesting

If you later add smarter tensor transforms, byte-plane separation, entropy coding, or model-specific packing, this same manifest structure can carry them.

## What the Qwen template showed
The Qwen template benchmark on repo assets produced exact round trips for every tested file.
Some files compressed a lot; some did not.
That variation is the whole point: **the system now has enough granularity to stop wasting space on things that should stay raw**.

## What the paper should say honestly
- exact round-trip is easy to state but hard to improve
- not all LLM assets are equally compressible
- metadata and tokenizer assets can compress extremely well
- weight shards need a more specialised approach
- universal 2x lossless compression is a goal, not a claim

## What could be done next
### Near-term engineering
- Add support for real safetensors shard parsing and chunking.
- Add per-block adaptive block sizes.
- Add model-tree manifests with dependency-aware restore order.
- Add streaming decompression so only the needed blocks are expanded.
- Add a better comparison harness against standard archive compressors.

### Research directions
- Tensor-aware canonicalisation before entropy coding.
- Byte-plane separation for floating-point weights.
- Residual coding after blockwise transforms.
- Shared scale tables across layers.
- Model families trained to be more compressible.
- Layer-by-layer streaming decode so only one layer is in memory at once.

### On the “invent a new algorithm” request
The current work is already a new algorithmic direction in the practical sense:
**MiddleOut Lattice** = blockwise file- and tensor-archive compression with per-block codec selection, raw fallback, and manifest-backed exact reconstruction.

It is not a theoretical breakthrough by itself, but it is a real architectural proposal that can be extended into one.

## Bottom line
The repo currently proves a useful, honest thing:
- exact lossless round-trip compression works
- file-local and block-local selection works
- Qwen repo assets can be benchmarked as a template
- the system is ready for deeper model-file research

It does **not** yet prove a universal 2x lossless compressor for all LLM weights. That remains a research problem.
