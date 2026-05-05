from pathlib import Path

from middleout_lattice.tensor_shards import (
    TensorArray,
    benchmark_tensor_strategies,
    build_safetensors_blob,
    compress_safetensors_blob,
    decompress_safetensors_blob,
)


def test_tensor_shard_roundtrip():
    tensors = [
        TensorArray(name='a', dtype='F16', shape=(8,), data=(b'abcd' * 4)),
        TensorArray(name='b', dtype='U8', shape=(16,), data=(b'\x00\x01' * 8)),
    ]
    blob = build_safetensors_blob(tensors)
    packed = compress_safetensors_blob(blob)
    restored = decompress_safetensors_blob(packed)
    assert restored == blob


def test_tensor_benchmark_has_mosaic_row():
    tensors = [
        TensorArray(name='a', dtype='F16', shape=(8,), data=(b'abcd' * 16)),
        TensorArray(name='b', dtype='U8', shape=(16,), data=(b'\x00\x01' * 16)),
    ]
    blob = build_safetensors_blob(tensors)
    rows = benchmark_tensor_strategies(blob)
    strategies = {row['strategy'] for row in rows}
    assert 'per-tensor mosaic' in strategies
    assert any(row['roundtrip_ok'] for row in rows if row['strategy'] == 'per-tensor mosaic')
