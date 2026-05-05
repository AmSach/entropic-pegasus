from pathlib import Path

from middleout_lattice import compress_directory, decompress_directory, compress_file_bytes, decompress_file_bytes
from middleout_lattice.codec import compress_bytes, decompress_bytes, roundtrip_path


def test_roundtrip_bytes():
    payload = b'abc' * 1000
    packed = compress_bytes(payload)
    assert decompress_bytes(packed.compressed_bytes) == payload


def test_roundtrip_path(tmp_path: Path):
    src = tmp_path / 'x.bin'
    src.write_bytes(b'hello' * 1000)
    result = roundtrip_path(src, tmp_path)
    assert result['roundtrip_ok'] is True
    assert (tmp_path / 'x.bin.molm').exists()


def test_archive_roundtrip(tmp_path: Path):
    src_dir = tmp_path / 'src'
    src_dir.mkdir()
    (src_dir / 'a.txt').write_text('hello world ' * 100)
    (src_dir / 'b.bin').write_bytes(b'\x00' * 1000)
    manifest = compress_directory(src_dir, tmp_path / 'packed', block_size=128)
    restored = decompress_directory(manifest, tmp_path / 'restored')
    assert (restored / 'a.txt').read_text() == (src_dir / 'a.txt').read_text()
    assert (restored / 'b.bin').read_bytes() == (src_dir / 'b.bin').read_bytes()
