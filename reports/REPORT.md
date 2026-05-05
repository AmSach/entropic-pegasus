# Benchmark report

- best file: tokenizer_config.json
- best codec: zlib
- original bytes: 7228
- compressed bytes: 1320
- ratio: 5.4758
- roundtrip ok: True

## Full comparison

- config.json / zlib: 1.4337x
- config.json / zlib: 1.4337x
- config.json / bz2: 1.3147x
- config.json / lzma: 1.2404x
- generation_config.json / zlib: 0.6079x
- generation_config.json / zlib: 0.6079x
- generation_config.json / bz2: 0.5391x
- generation_config.json / lzma: 0.4842x
- merges.txt / bz2: 3.1819x
- merges.txt / bz2: 3.1819x
- merges.txt / lzma: 2.8953x
- merges.txt / zlib: 2.3726x
- tokenizer.json / lzma: 5.1005x
- tokenizer.json / lzma: 5.1005x
- tokenizer.json / bz2: 4.9728x
- tokenizer.json / zlib: 3.6551x
- tokenizer_config.json / zlib: 5.4758x
- tokenizer_config.json / zlib: 5.4758x
- tokenizer_config.json / lzma: 5.2759x
- tokenizer_config.json / bz2: 5.0758x
- vocab.json / lzma: 3.6951x
- vocab.json / lzma: 3.6951x
- vocab.json / bz2: 3.2207x
- vocab.json / zlib: 2.5941x
