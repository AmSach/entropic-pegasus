from pathlib import Path

from middleout_lattice.codec import compress_bytes, decompress_bytes, roundtrip_path


def test_roundtrip_bytes():
    payload = b"abc" * 1000
    packed = compress_bytes(payload)
    assert decompress_bytes(packed.compressed_bytes) == payload


def test_roundtrip_path(tmp_path: Path):
    src = tmp_path / "x.bin"
    src.write_bytes(b"hello" * 1000)
    result = roundtrip_path(src, tmp_path)
    assert result["roundtrip_ok"] is True
    assert (tmp_path / "x.bin.molm").exists()
    assert (tmp_path / "x.bin.restored").exists()
