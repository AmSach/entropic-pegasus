from pathlib import Path

from middleout_lattice import CompressedModelStore, compress_directory, decompress_directory
from middleout_lattice.archive import compress_file_bytes, decompress_file_bytes
from middleout_lattice.codec import compress_bytes, decompress_bytes, roundtrip_path
from middleout_lattice.mosaic import mosaic_encode, mosaic_decode


def test_roundtrip_bytes():
    payload = b'abc' * 1000
    packed = compress_bytes(payload)
    assert decompress_bytes(packed.compressed_bytes) == payload


def test_mosaic_roundtrip():
    payload = bytes(range(256)) * 8
    encoded, meta = mosaic_encode(payload, word_size=4)
    assert mosaic_decode(encoded, meta) == payload


def test_archive_roundtrip(tmp_path: Path):
    src_dir = tmp_path / 'src'
    src_dir.mkdir()
    (src_dir / 'a.txt').write_text('hello world ' * 100)
    (src_dir / 'b.bin').write_bytes(b'\x00' * 1000)
    archive, record = compress_file_bytes((src_dir / 'a.txt').read_bytes(), block_size=64)
    assert decompress_file_bytes(archive, record) == (src_dir / 'a.txt').read_bytes()
    manifest = compress_directory(src_dir, tmp_path / 'packed', block_size=128)
    restored = decompress_directory(manifest, tmp_path / 'restored')
    assert (restored / 'a.txt').read_text() == (src_dir / 'a.txt').read_text()
    assert (restored / 'b.bin').read_bytes() == (src_dir / 'b.bin').read_bytes()


def test_model_store_roundtrip(tmp_path: Path):
    src_dir = tmp_path / 'src2'
    src_dir.mkdir()
    (src_dir / 'config.json').write_text('{"a": 1, "b": 2}')
    (src_dir / 'weights.bin').write_bytes((b'abcd' * 1000) + b'\x00' * 2048)
    store = CompressedModelStore.from_source(src_dir, tmp_path / 'packed2', block_size=256)
    assert store.read_bytes('config.json') == (src_dir / 'config.json').read_bytes()
    assert store.read_bytes('weights.bin') == (src_dir / 'weights.bin').read_bytes()
    with store.open_materialized() as mat:
        assert (mat / 'config.json').read_bytes() == (src_dir / 'config.json').read_bytes()
        assert (mat / 'weights.bin').read_bytes() == (src_dir / 'weights.bin').read_bytes()
