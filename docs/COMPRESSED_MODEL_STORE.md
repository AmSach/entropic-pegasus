# Compressed model store runtime

This repo now has a runtime layer for keeping model files compressed on disk while still letting the program read them as normal bytes when needed.

## The idea
Instead of unpacking a whole model directory eagerly, the store:
- compresses each file independently
- keeps a manifest with hashes, sizes, and storage paths
- decompresses only the file you ask for
- can materialise a temp copy when a downstream library insists on ordinary files

## Why this is useful
Most model repos are a mix of:
- big binary weight shards
- tokenizer and vocab JSON
- config files
- metadata

Those files do not behave the same. Some compress well, some do not. So the runtime should not be stupid and force one policy everywhere.

## API
The main class is `CompressedModelStore` in `file 'middleout_lattice/model_store.py'`.

### Create a store
```python
from pathlib import Path
from middleout_lattice import CompressedModelStore

store = CompressedModelStore.from_source(
    Path("/path/to/model_repo"),
    Path("/path/to/compressed_store"),
    block_size=1 << 20,
)
```

### Read a file back
```python
config_bytes = store.read_bytes("config.json")
```

### Materialise a working copy
```python
with store.open_materialized() as mounted:
    # `mounted` is a temp directory with the original file tree restored
    print((mounted / "config.json").read_text())
```

## Exactness
The store is lossless. It stores:
- raw or compressed blocks
- SHA-256 hashes for blocks and files
- the original byte sizes
- the storage location for each file

If any byte changes, decode fails.

## Skip-when-bigger rule
If compression makes a file larger, the store keeps it raw. That is the sane default and it prevents fake wins.

## What this does not yet do
- It does not magically stream a transformer layer-by-layer from a compressed shard.
- It does not beat entropy on already-random data.
- It does not solve every model format in the wild.

## Next step
The next real research step is to make the store aware of tensor structure:
- weight shard parsing
- blockwise tensor packing
- exponent/mantissa separation
- residual coding
- streaming decode for large checkpoints
