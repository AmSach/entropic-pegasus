# Paper outline: lossless compression for whole-model LLM artefacts

This is the skeleton for a longer paper. The actual manuscript should be expanded into a full 30+ page submission with proofs, derivations, ablations, and large-scale experiments.

## Working title
**MiddleOut Lattice: Lossless Compression of LLM Artefacts with Codec-Aware Tensor Packing**

## Abstract draft
We study exact, lossless compression of model artefacts for large language models. The central question is not whether one can quantise a cache, but whether one can reduce the storage and transfer cost of the model itself while preserving exact reconstruction. We present a codec-first architecture that compresses arbitrary model files, evaluates codec choice per file, and serves as a template for tensor-aware compression. Experiments on Qwen2.5-0.5B artefacts show that some model files compress substantially while remaining exactly recoverable, but also demonstrate that file type strongly controls compressibility. We discuss the limits of general-purpose codecs, derive the relevant information-theoretic constraints, and outline a path toward architecture-aware compression of weight tensors.

## Suggested section structure
1. Introduction
2. Problem statement
3. Background and related work
4. Information-theoretic limits
5. Codec-first architecture
6. Byte-level archive format
7. Exact reconstruction guarantees
8. Qwen template benchmark
9. Results and analysis
10. Model-agnostic generalisation
11. Towards tensor-aware codecs
12. Lossless compression of floating-point weights
13. Entropy coding and residual coding
14. Streaming decode and memory control
15. Open problems
16. Limitations
17. Conclusion

## Mathematical topics to cover
- entropy and cross-entropy
- Shannon source coding theorem
- ideal code length bounds
- redundancy in structured tensors
- byte-plane decomposition of floats
- blockwise transform coding
- error-free reconstruction constraints
- compression ratio definitions
- memory footprint vs storage footprint
- decode-time complexity

## Candidate derivations
### 1. Compression ratio
For original size `S_o` and compressed size `S_c`:

```latex
R = \\frac{S_o}{S_c}
```

### 2. Entropy lower bound
For symbol distribution `p`:

```latex
H(p) = -\\sum_i p_i \\log_2 p_i
```

Any lossless compressor must satisfy the usual lower bound on expected code length:

```latex
\\mathbb{E}[L] \\ge H(p)
```

### 3. Byte-plane decomposition
If a floating-point value is decomposed into sign, exponent, and mantissa components, then the compression problem can be written as a separate coding problem per substream.

### 4. Blockwise coding
For tensor block `B_k`, a codec may use different local models or codebooks:

```latex
L(B) = \\sum_k L(B_k)
```

This is the starting point for a more practical architecture-aware compressor.

## Experiments to include
- Qwen2.5-0.5B file-level benchmark
- per-file codec comparison
- full-weight shard benchmark on one or more models
- block size ablation
- codec family ablation
- tensor format ablation
- speed vs ratio curves

## Figures to include
- benchmark bar chart
- compression ratio histogram
- file-type vs compressibility scatter plot
- architecture diagram of encode/decode flow
- memory-use diagram for streaming decode

## What the paper should say honestly
- exact round-trip is easy to state but hard to improve
- not all LLM assets are equally compressible
- metadata and tokenizer assets can compress extremely well
- weight shards need a more specialised approach
- universal 2x lossless compression is a goal, not a claim

## Next writing step
The next step is to turn this outline into a full manuscript with:
- formal definitions
- theorem statements where possible
- proofs or proof sketches
- tables of results
- a limitations section that does not lie
