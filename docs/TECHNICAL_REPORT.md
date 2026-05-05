# Technical report: MiddleOut Lattice

## What we built
MiddleOut Lattice is a lossless compression prototype for model files and tensor archives. The important point is that it is **exact**: whatever goes in must come back byte-for-byte.

The repo was shaped around a simple reality:
- KV-cache compression is not whole-model compression.
- If you want whole-model lossless reduction, you need to compress the **files that define the model**.
- The most honest first target is the on-disk model assets and metadata files used by a model repository.

## What I changed
### 1. Created a new repo
I created `entropic-pegasus` on GitHub and pushed the code there.

### 2. Built a codec-first archive layer
The core module lives in `middleout_lattice/codec.py`. It does four things:
- compresses raw bytes
- stores a checksum and size header
- chooses the smallest of a few codecs
- restores bytes exactly on decode

The current codecs are:
- `lzma`
- `zlib`
- `bz2`

For each input blob, the code computes:
- `sha256(original_bytes)`
- original size
- encoded payload size
- decoded integrity check

The archive format is intentionally boring and explicit:
- magic prefix
- JSON header
- encoded payload

That makes it easy to debug and hard to lie to yourself about.

### 3. Added benchmark tooling
I added `scripts/benchmark_model.py`, which downloads selected files from a Hugging Face repo and compares codecs per file. For the Qwen template run, it used these files:
- `config.json`
- `generation_config.json`
- `merges.txt`
- `tokenizer.json`
- `tokenizer_config.json`
- `vocab.json`

The benchmark records:
- file name
- codec used
- original bytes
- compressed bytes
- compression ratio
- exact round-trip success

### 4. Added reporting and charting
I added `scripts/make_report.py` and `scripts/render_report_png.py` to generate:
- `reports/REPORT.md`
- `reports/comparison.svg`
- `reports/comparison.png`

### 5. Added tests
The repo includes round-trip tests that verify:
- a small byte payload survives compression and decompression exactly
- a path round-trip writes a recovered file that matches the source

## How it works
### Lossless archive encoding
For a blob of bytes `x`, the encoder produces:

```text
magic || header_length || payload_length || header || payload
```

where:
- `header` stores the algorithm name, SHA-256, and size
- `payload` is the compressed content

On decode, the system:
1. parses the header
2. decompresses the payload
3. recomputes SHA-256
4. compares hashes
5. checks the original size

If anything differs, decode fails.

### Why this is model-agnostic
This approach does not care whether the bytes came from:
- a tokenizer file
- a config file
- a safetensors shard
- a binary weight blob

If it is bytes, it can be tested. That does not mean it will compress well. It means the pipeline is general.

### Why the Qwen result is a template, not a victory lap
The benchmark on Qwen repo assets is a **template test**, not a proof that all frozen LLM weights will get 2x lossless compression.

The measured gains in the report came from smaller text/tokenizer/config assets, where general-purpose lossless codecs can do very well. That is useful, but it is not the same as proving compression on the giant weight tensors themselves.

## What the Qwen template data showed
The Qwen template benchmark produced exact round trips for every tested file.

Best observed file in the report:
- `tokenizer_config.json`
- codec: `zlib`
- ratio: `5.4758x`

Other files varied a lot:
- some compressed well
- some barely compressed
- one or two got larger than the original under a particular codec

That variation is exactly what you expect when you stop pretending all file types are the same.

## What this means technically
### Good news
- Exact decode is straightforward.
- File-level compression and benchmarking are already working.
- The repo can be extended to any model family by swapping in a different file list.

### Bad news
- Universal 2x lossless compression for arbitrary LLM weights is not guaranteed.
- Some tensor formats are already close to entropy-limited.
- Real weight compression likely needs architecture-aware codecs, blockwise packing, and possibly model design changes.

## What could be done next
### Near-term engineering
- Add support for more model file types.
- Add proper safetensors parsing.
- Benchmark full weight shards, not just metadata files.
- Add a reproducible CLI that accepts any Hugging Face repo.
- Add smarter codec selection per file type and per block.

### Research directions
- Tensor-aware canonicalisation before entropy coding.
- Byte-plane separation for floating-point weights.
- Blockwise residual coding.
- Shared scale tables across layers.
- Model families trained to be more compressible.
- Layer-by-layer streaming decode so only one layer is in memory at once.

### Publication path
To turn this into an arXiv/OpenReview paper, the next stage should be:
1. a formal problem statement
2. compression theory and bounds
3. codec design
4. empirical evaluation
5. ablation studies
6. limitations and failure cases

## Bottom line
The repo currently proves a useful, honest thing:
- exact lossless round-trip compression works
- Qwen repo assets can be benchmarked as a template
- the system is ready for deeper model-file research

It does **not** yet prove a universal 2x lossless compressor for all LLM weights. That is still a research problem, not a solved product.